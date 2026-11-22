"""
Runner Utama Eksperimen Skripsi: Pengujian Faktorial Multi-Kondisi pada Anvil.

Mengeksekusi matriks pengujian sesuai Bab III Metode Penelitian:
  - Kelompok Kontrak Target (Sub-bab 3.4):
      (a) Controlled Synthetic Benchmark (VulnerableVault)
      (b) Real-World Smart Contracts SC-Bench (Shallow, Medium, Deep)
  - Dinamika Harga Gas Jaringan (Sub-bab 3.2.1 & 3.5):
      Rezim 1 (Stable) vs Rezim 2 (Volatile Spikes)
  - Variasi Anggaran Modal Gas B (Sub-bab 3.5):
      Tight (B_low), Moderate (B_med), Loose (B_high)
  - Lima Kebijakan Uji & Ablasi (Sub-bab 3.3.5):
      1. UniformRandom (Forge Fuzz Baseline)
      2. LinUCB (Stationary Unconstrained: gamma = 1, lambda_t = 0)
      3. D-LinUCB (Discounted Unconstrained: gamma in (0,1), lambda_t = 0)
      4. LinCBwK (Stationary Knapsack: gamma = 1, dynamic lambda_t)
      5. D-LinCBwK (Discounted Knapsack - Proposed: gamma in (0,1), dynamic lambda_t)
  - Lima Metrik Evaluasi Kinerja & Uji Statistik (Sub-bab 3.6):
      1. Cumulative Dynamic Regret D-Regret(T) (Persamaan 3.22)
      2. Frekuensi Pemicuan Kerentanan (E4), Mutasi Status (SSTORE & Waypoint), Imbalan Kumulatif
      3. Rasio Efisiensi Penggunaan Anggaran Gas Eff_gas (Persamaan 3.23)
      4. Tingkat Kelangsungan Hidup Anggaran SR = tau / T
      5. Waktu Komputasi Keputusan (Execution Overhead ms) & Uji Signifikansi (Paired t-test & Wilcoxon)
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

# Ensure module path imports
CURR_DIR = Path(__file__).resolve().parent
CMAB_ROOT = CURR_DIR.parent
REPO_ROOT = CMAB_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CMAB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CMAB_ROOT / "src"))
if str(CMAB_ROOT) not in sys.path:
    sys.path.insert(0, str(CMAB_ROOT))

from cmab_anvil.algorithms.linucb import DisjointLinUCB
from cmab_anvil.algorithms.discounted_linucb import DiscountedLinUCB
from cmab_anvil.algorithms.uniform_random import UniformRandomPolicy
from cmab_anvil.algorithms.linucb_knapsack import StationaryLinCBwK
from cmab_anvil.algorithms.discounted_linucb_knapsack import DiscountedLinCBwK
from cmab_anvil.algorithms.dynamic_regret import compare_policies_dynamic_regret
from cmab_anvil.common.anvil_rpc import (
    UnprivilegedAnvilClient,
    keccak256_bytes,
    compute_solidity_mapping_slot,
)
from cmab_anvil.Agen_Vault_Custody.context import (
    PublicContextExtractor,
    OnChainContextSnapshot,
    CONTEXT_DIM,
)
from cmab_anvil.Agen_Vault_Custody.reward import (
    VaultCustodyRewardOracle,
    BanditRewardBreakdown,
)

from cmab_anvil.experiments_skripsi.config import (
    SOLC_BIN,
    CAST_BIN,
    ANVIL_BIN,
    RPC_URL,
    DEPLOYER_ADDR,
    DEPLOYER_PK,
    PEER_ADDR,
    PEER_PK,
    AGENT_ADDR,
    AGENT_PK,
    MOCK_TOKEN_SOL,
    VULNERABLE_VAULT_SOL,
    DATASET_DIR,
    TELEMETRY_DIR,
    BUDGET_TIERS,
    DEFAULT_HORIZON_T,
    ALPHA,
    GAMMA,
    L2_LAMBDA,
    INITIAL_LAMBDA,
    MIN_CAPITAL_THRESHOLD_ETH,
)
from cmab_anvil.experiments_skripsi.gas_regimes import generate_gas_series
from cmab_anvil.experiments_skripsi.contract_catalog import (
    CATALOG_CONTRACTS,
    get_contracts_by_tier,
)


COMPILED_ABIS: Dict[str, List[Dict[str, Any]]] = {}
GAS_USED_REGEX = re.compile(r"gasUsed\s+(\d+|0x[0-9a-fA-F]+)")
REWARD_ORACLE = VaultCustodyRewardOracle()


def ensure_anvil_running() -> None:
    p = subprocess.run(
        [CAST_BIN, "block-number", "--rpc-url", RPC_URL],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        print("[*] Menjalankan instance Anvil di port 8545 (--steps-tracing aktif)...")
        subprocess.Popen(
            [
                ANVIL_BIN,
                "--port", "8545",
                "--chain-id", "31337",
                "--accounts", "10",
                "--balance", "10000",
                "--gas-limit", "30000000",
                "--steps-tracing",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(1.5)


def run_cast(args: List[str]) -> Tuple[bool, str, str]:
    if args[0] in ("keccak", "abi-encode"):
        cmd = [CAST_BIN] + args
    else:
        cmd = [CAST_BIN, args[0], "--rpc-url", RPC_URL] + args[1:]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return (p.returncode == 0), p.stdout.strip(), p.stderr.strip()


def rpc_call(method: str, params: List[Any]) -> Any:
    payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 1})
    p = subprocess.run(
        ["curl", "-s", "-X", "POST", "-H", "Content-Type: application/json", "--data", payload, RPC_URL],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        return json.loads(p.stdout).get("result")
    except Exception:
        return None


def compile_contract(fpath: Path, cname: str) -> Optional[str]:
    res = subprocess.run(
        [SOLC_BIN, "--combined-json", "abi,bin", str(fpath)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if res.returncode != 0:
        return None
    idx = res.stdout.find("{")
    if idx == -1:
        return None
    data = json.loads(res.stdout[idx:])
    for k, v in data.get("contracts", {}).items():
        if k.endswith(":" + cname):
            raw_abi = v.get("abi")
            if isinstance(raw_abi, str):
                try:
                    COMPILED_ABIS[cname] = json.loads(raw_abi)
                except Exception:
                    pass
            elif isinstance(raw_abi, list):
                COMPILED_ABIS[cname] = raw_abi
            return v.get("bin")
    return None


def deploy_bytecode(
    bytecode: str,
    constructor_sig: Optional[str] = None,
    constructor_args: Optional[List[str]] = None,
) -> Optional[str]:
    create_hex = "0x" + bytecode
    if constructor_sig and constructor_args:
        ok, encoded, _ = run_cast(["abi-encode", constructor_sig] + constructor_args)
        if ok and encoded.startswith("0x"):
            create_hex += encoded[2:]
    rpc_call("anvil_setNextBlockBaseFeePerGas", ["0x3b9aca00"])  # 1 Gwei
    ok, out, _ = run_cast(
        ["send", "--private-key", DEPLOYER_PK, "--gas-price", "5000000000", "--create", create_hex]
    )
    if not ok:
        return None
    for line in out.splitlines():
        if "contractAddress" in line:
            return line.split()[-1]
    return None


def send_tx(
    pk: str,
    to_addr: str,
    sig: str = "",
    args: Optional[List[str]] = None,
    eth_val: float = 0.0,
    gas_price_gwei: Optional[float] = None,
) -> Tuple[bool, str]:
    cmd = ["send", "--private-key", pk]
    if gas_price_gwei is not None and gas_price_gwei > 0:
        cmd.extend(["--gas-price", f"{int(gas_price_gwei * 1e9)}"])
    if eth_val > 0:
        cmd.extend(["--value", f"{eth_val:.6f}ether"])
    cmd.append(to_addr)
    if sig:
        cmd.append(sig)
        if args:
            cmd.extend(args)
    ok, out, err = run_cast(cmd)
    return ok, (out if ok else err)


def parse_gas_used(tx_ok: bool, tx_out: str, n_args: int = 0) -> int:
    """
    Mengekstrak GasUsed_t(a_t) dari keluaran transaksi EVM sesuai Persamaan (3.11).
    Jika transaksi revert pada tahap guard EVM, menggunakan estimasi gas intrinsik + evaluasi calldata.
    """
    if tx_ok and tx_out:
        m = GAS_USED_REGEX.search(tx_out)
        if m:
            raw_val = m.group(1)
            try:
                return max(21000, int(raw_val, 0))
            except ValueError:
                pass
        return 55000 + 4000 * n_args
    return 36000 + 5000 * n_args


def compute_single_node_merkle_root(account_addr: str, amount_wei: int) -> str:
    """Menghitung keccak256(abi.encodePacked(address, uint256)) untuk inisialisasi Airdrop."""
    addr_bytes = bytes.fromhex(account_addr.lower().replace("0x", ""))
    amt_bytes = int(amount_wei).to_bytes(32, "big")
    return "0x" + keccak256_bytes(addr_bytes + amt_bytes).hex()


def setup_member_environment(
    spec: Dict[str, Any],
    caddr: str,
    mock_token_addr: str,
    client: UnprivilegedAnvilClient,
) -> Dict[str, Any]:
    """
    Inisialisasi lingkungan kontrak sebelum horizon pengujian dimulai (Sub-bab 3.5 Tahap 1 & 2).
    Mengembalikan state pelacakan invarian (B_t^(C), L_t^(C), Omega_t) untuk verifikasi orakel Persamaan (3.4).
    """
    cid = spec["id"]
    MAX_UINT256 = "115792089237316195423570985008687907853269984665640564039457584007913129639935"

    # Alokasi token utilitas standar pada MockToken
    send_tx(DEPLOYER_PK, mock_token_addr, "mint(address,uint256)", [AGENT_ADDR, "100000000000000000000000"])
    send_tx(DEPLOYER_PK, mock_token_addr, "mint(address,uint256)", [PEER_ADDR, "100000000000000000000000"])
    send_tx(DEPLOYER_PK, mock_token_addr, "mint(address,uint256)", [caddr, "100000000000000000000000"])
    send_tx(AGENT_PK, mock_token_addr, "approve(address,uint256)", [caddr, MAX_UINT256])
    send_tx(PEER_PK, mock_token_addr, "approve(address,uint256)", [caddr, MAX_UINT256])

    custody_token_addr = mock_token_addr

    if "B01" in cid:
        # VulnerableVault: ambang emergencyWithdraw adalah >= 5.00 ETH.
        # Inisialisasi dengan 4.48 ETH (Deployer) + 0.50 ETH (Peer) = 4.98 ETH (0.02 ETH di bawah ambang).
        send_tx(DEPLOYER_PK, caddr, "deposit()", [], 4.48)
        send_tx(PEER_PK, caddr, "deposit()", [], 0.50)
    elif "S01" in cid:
        custody_token_addr = caddr
        send_tx(DEPLOYER_PK, caddr, "enableTrading()", [])
        send_tx(
            DEPLOYER_PK,
            caddr,
            "setDepositMinMax(uint256,uint256)",
            ["10000000000000000", "100000000000000000000"],
        )
        send_tx(DEPLOYER_PK, caddr, "transfer(address,uint256)", [AGENT_ADDR, "50000000000000000000"])
        send_tx(DEPLOYER_PK, caddr, "transfer(address,uint256)", [PEER_ADDR, "50000000000000000000"])
        send_tx(DEPLOYER_PK, caddr, "transfer(address,uint256)", [caddr, "10000000000000000000"])
        send_tx(AGENT_PK, caddr, "approve(address,uint256)", [caddr, MAX_UINT256])
        send_tx(PEER_PK, caddr, "approve(address,uint256)", [caddr, MAX_UINT256])
        send_tx(DEPLOYER_PK, caddr, "", [], 1.0)
        send_tx(PEER_PK, caddr, "", [], 0.50)
    elif "S02" in cid:
        custody_token_addr = caddr
        # Alokasikan saldo awal token pada slot mapping _balances (slot 1) untuk AGENT_ADDR dan PEER_ADDR
        agent_bal_slot = compute_solidity_mapping_slot(AGENT_ADDR, 1)
        peer_bal_slot = compute_solidity_mapping_slot(PEER_ADDR, 1)
        init_tok_hex = "0x" + f"{1000 * 10**18:064x}"
        rpc_call("anvil_setStorageAt", [caddr, agent_bal_slot, init_tok_hex])
        rpc_call("anvil_setStorageAt", [caddr, peer_bal_slot, init_tok_hex])
        send_tx(AGENT_PK, caddr, "approve(address,uint256)", [caddr, MAX_UINT256])
    elif "M01" in cid:
        send_tx(DEPLOYER_PK, caddr, "setCost(uint256)", ["10000000000000000"])
        # Owner mencetak URI statis dengan 0 ETH agar escrow mulai dari 0 ETH
        send_tx(DEPLOYER_PK, caddr, "mint(string)", ["STATIC_DUPLICATE"], 0.0)
    elif "M02" in cid:
        # Inisialisasi Room 1 dan Room 2 pada Shuffle (minBid = 0.01 ETH, maxBid = 5.0 ETH)
        send_tx(
            DEPLOYER_PK,
            caddr,
            "createRoom(string,uint256,uint256,uint256,uint256,uint256,address)",
            ["Room1", "10000000000000000", "5000000000000000000", "5", "10", "3600", DEPLOYER_ADDR],
        )
        send_tx(
            DEPLOYER_PK,
            caddr,
            "createRoom(string,uint256,uint256,uint256,uint256,uint256,address)",
            ["Room2", "10000000000000000", "5000000000000000000", "5", "10", "3600", DEPLOYER_ADDR],
        )
    elif "D01" in cid:
        custody_token_addr = caddr
        send_tx(DEPLOYER_PK, caddr, "transfer(address,uint256)", [AGENT_ADDR, "5000000000000000000000"])
        send_tx(DEPLOYER_PK, caddr, "transfer(address,uint256)", [PEER_ADDR, "5000000000000000000000"])
        send_tx(DEPLOYER_PK, caddr, "transfer(address,uint256)", [caddr, "100000000000000000000"])
        send_tx(AGENT_PK, caddr, "approve(address,uint256)", [caddr, MAX_UINT256])
        send_tx(PEER_PK, caddr, "approve(address,uint256)", [caddr, MAX_UINT256])
    elif "D02" in cid:
        send_tx(DEPLOYER_PK, caddr, "initialize(address,address)", [mock_token_addr, DEPLOYER_ADDR])
        send_tx(DEPLOYER_PK, caddr, "setAllowStaking(bool)", ["true"])
        send_tx(DEPLOYER_PK, caddr, "setAllowClaim(bool)", ["true"])
        send_tx(DEPLOYER_PK, caddr, "setReward(uint256,uint256)", ["1", "5000"])
        send_tx(DEPLOYER_PK, caddr, "setReward(uint256,uint256)", ["7", "8000"])
    elif "D03" in cid:
        send_tx(DEPLOYER_PK, caddr, "setToken(address)", [mock_token_addr])
        merkle_root = compute_single_node_merkle_root(AGENT_ADDR, 250 * 10**18)
        send_tx(DEPLOYER_PK, caddr, "setRoot(bytes32)", [merkle_root])

    init_eth = client.get_eth_balance(caddr)
    init_escrow = get_escrow_balance(caddr, client) if "M01" in cid else 0.0
    init_tok_res = client.call_uint256_ether_with_fallbacks(
        custody_token_addr, ["balanceOf(address)(uint256)"], caddr
    )

    return {
        "cid": cid,
        "custody_token_addr": custody_token_addr,
        "eth_liability": init_eth + init_escrow,
        "tok_liability": init_tok_res,
        "peer_deposited_eth": 0.50 if ("B01" in cid or "S01" in cid) else 0.0,
    }


def get_escrow_balance(caddr: str, client: UnprivilegedAnvilClient) -> float:
    """Membaca saldo PullPayment escrow untuk M01 (DrawTT)."""
    return client.call_uint256_ether_with_fallbacks(
        caddr, ["payments(address)(uint256)"], DEPLOYER_ADDR
    )


def check_shuffle_room2_intact(caddr: str) -> bool:
    """Memeriksa apakah konfigurasi fundsWallet pada Room 2 Shuffle belum dibajak oleh AGENT_ADDR."""
    ok, out, _ = run_cast(["call", caddr, "getRoom(uint256)", "2"])
    if not ok or not out:
        return True
    agent_clean = AGENT_ADDR.lower().replace("0x", "")
    return agent_clean not in out.lower()


def pregenerate_network_events(contract_idx: int, rounds: int, seed_offset: int = 0) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed=202600 + contract_idx * 37 + seed_offset * 1009)
    events = []
    for t in range(1, rounds + 1):
        ev_type = rng.choice(
            ["peer_liquidity", "peer_withdrawal", "block_jitter", "congestion_poke", "none"],
            p=[0.25, 0.25, 0.20, 0.15, 0.15],
        )
        eth_delta = float(np.round(rng.uniform(0.02, 0.08), 3))
        time_delta = int(rng.integers(3600, 864000))
        events.append({
            "round": t,
            "type": str(ev_type),
            "eth_delta": eth_delta,
            "time_delta": time_delta,
        })
    return events


def execute_network_event(
    ev: Dict[str, Any],
    caddr: str,
    mock_token_addr: str,
    env_state: Dict[str, Any],
    client: UnprivilegedAnvilClient,
) -> None:
    """
    Mengeksekusi aktivitas pengguna eksternal (PEER_PK) dan pergeseran waktu blok secara murni
    tanpa cheatcode anvil_impersonateAccount (Sub-bab 3.2.1 & 3.5).
    """
    cid = env_state["cid"]
    ev_type = ev["type"]

    if ev_type == "peer_liquidity":
        if "B01" in cid:
            # Jaga agar saldo VulnerableVault tetap sedikit di bawah 5.0 ETH sampai agen menyetor
            cur_eth = client.get_eth_balance(caddr)
            if cur_eth + ev["eth_delta"] < 4.99:
                ok, _ = send_tx(PEER_PK, caddr, "deposit()", [], ev["eth_delta"])
                if ok:
                    env_state["peer_deposited_eth"] += ev["eth_delta"]
                    env_state["eth_liability"] += ev["eth_delta"]
        elif "S01" in cid:
            ok, _ = send_tx(PEER_PK, caddr, "", [], ev["eth_delta"])
            if ok:
                env_state["peer_deposited_eth"] += ev["eth_delta"]
                env_state["eth_liability"] += ev["eth_delta"]
        else:
            send_tx(PEER_PK, mock_token_addr, "transfer(address,uint256)", [DEPLOYER_ADDR, "1000000000000000"])

    elif ev_type == "peer_withdrawal":
        # Transaksi penarikan sah oleh PEER_PK tanpa anvil_impersonateAccount
        if "B01" in cid and env_state.get("peer_deposited_eth", 0.0) >= ev["eth_delta"]:
            cur_eth = client.get_eth_balance(caddr)
            if cur_eth - ev["eth_delta"] >= 4.94:
                wei_str = str(int(round(ev["eth_delta"] * 1e18)))
                ok, _ = send_tx(PEER_PK, caddr, "withdraw(uint256)", [wei_str], 0.0)
                if ok:
                    env_state["peer_deposited_eth"] -= ev["eth_delta"]
                    env_state["eth_liability"] = max(0.0, env_state["eth_liability"] - ev["eth_delta"])
            else:
                send_tx(PEER_PK, mock_token_addr, "transfer(address,uint256)", [DEPLOYER_ADDR, "500000000000000"])
        else:
            send_tx(PEER_PK, mock_token_addr, "transfer(address,uint256)", [DEPLOYER_ADDR, "500000000000000"])

    elif ev_type == "block_jitter":
        rpc_call("evm_increaseTime", [ev["time_delta"]])
        rpc_call("evm_mine", [])

    elif ev_type == "congestion_poke":
        send_tx(PEER_PK, mock_token_addr, "transfer(address,uint256)", [DEPLOYER_ADDR, "1000000000000000"])


def build_context_vector(
    caddr: str,
    mock_token_addr: str,
    client: UnprivilegedAnvilClient,
    extractor: PublicContextExtractor,
    env_state: Dict[str, Any],
    recent_revert_rate: float,
    agent_budget_eth: float,
    sstore_executed: bool = False,
    dataflow_waypoint: bool = False,
    min_deposit_requirement: float = 0.05,
) -> Tuple[np.ndarray, OnChainContextSnapshot]:
    """
    Membangun vektor konteks ternormalisasi x_t in R^10 (Tabel 3.1) dan snapshot invarian on-chain.
    """
    cid = env_state["cid"]
    custody_tok_addr = env_state["custody_token_addr"]

    block = client.get_block_number()
    vault_eth = client.get_eth_balance(caddr)
    escrow_eth = get_escrow_balance(caddr, client) if "M01" in cid else 0.0
    effective_vault_eth = vault_eth + escrow_eth

    code_alive = client.get_code_alive(caddr)
    if "M02" in cid and code_alive:
        code_alive = check_shuffle_room2_intact(caddr)

    agent_bal = client.query_agent_stake(caddr, AGENT_ADDR, fallback_token_address=custody_tok_addr)
    token_bal = client.call_uint256_ether_with_fallbacks(
        custody_tok_addr, ["balanceOf(address)(uint256)"], AGENT_ADDR
    )
    contract_tok_res = client.call_uint256_ether_with_fallbacks(
        custody_tok_addr, ["balanceOf(address)(uint256)"], caddr
    )

    total_dep = float(env_state["eth_liability"])
    eth_solvent = bool(effective_vault_eth + 1e-6 >= total_dep)
    tok_solvent = bool(contract_tok_res + 1e-6 >= float(env_state["tok_liability"]))
    is_solvent = bool(eth_solvent and tok_solvent)

    lock_ok, locked_val = client.call_bool(caddr, "isLocked()(bool)")
    is_locked = locked_val if lock_ok else False

    snap = OnChainContextSnapshot(
        block_number=block,
        vault_eth_balance=effective_vault_eth,
        vault_total_deposited=total_dep,
        agent_vault_balance=agent_bal,
        agent_eoa_eth_balance=agent_budget_eth,
        agent_token_balance=token_bal,
        vault_is_solvent=is_solvent,
        vault_is_locked=is_locked,
        vault_code_alive=code_alive,
        recent_revert_rate=recent_revert_rate,
        sstore_executed=sstore_executed,
        dataflow_waypoint=dataflow_waypoint,
        min_deposit_requirement=min_deposit_requirement,
    )
    vec = extractor.to_vector(snap)
    return vec, snap


def execute_member_arm(
    spec: Dict[str, Any],
    arm: Dict[str, Any],
    caddr: str,
    mock_token_addr: str,
    round_t: int,
    gas_price_gwei: Optional[float] = None,
) -> Tuple[bool, str, List[str]]:
    target = mock_token_addr if arm.get("target") == "MOCK_TOKEN" else caddr
    sig = arm["sig"]
    val = float(arm.get("val", 0.0))

    resolved_args: List[str] = []
    for a in arm["args"]:
        if a == "DYNAMIC_URI":
            resolved_args.append(f"member_nft_{round_t}")
        elif a == "DYNAMIC_URI_LOW":
            resolved_args.append(f"member_low_{round_t}")
        elif a == "DYNAMIC_SIG65":
            # 65-byte signature dengan r unik per putaran dan v = 0 (ecrecover -> address(0))
            resolved_args.append("0x" + f"{round_t:064x}" + ("00" * 33))
        elif a == "DYNAMIC_STAKE_IDX":
            ok_st, out_st, _ = run_cast(["call", caddr, "getTotalStakes()(uint256)"])
            idx_val = 0
            if ok_st and out_st:
                try:
                    total_st = int(out_st.split()[0], 0)
                    idx_val = max(0, total_st - 1)
                except ValueError:
                    idx_val = 0
            resolved_args.append(str(idx_val))
        elif a == "DYNAMIC_ID":
            resolved_args.append(str(100 + round_t))
        elif a == "DYNAMIC_ID_ERC20":
            resolved_args.append(str(200 + round_t))
        elif a == "DYNAMIC_ENDTIME":
            resolved_args.append(str(2500000000 + round_t * 10000))
        elif a == "MOCK_TOKEN":
            resolved_args.append(mock_token_addr)
        else:
            resolved_args.append(a)

    ok, out = send_tx(AGENT_PK, target, sig, resolved_args, val, gas_price_gwei=gas_price_gwei)
    return ok, out, resolved_args


def update_legitimate_liabilities_before_post_snapshot(
    spec: Dict[str, Any],
    arm: Dict[str, Any],
    tx_ok: bool,
    caddr: str,
    env_state: Dict[str, Any],
    client: UnprivilegedAnvilClient,
) -> None:
    """
    Memperbarui kewajiban simpanan sah L_{t+1}^(C) ketika transaksi setoran/staking normal berhasil,
    sehingga penarikan normal yang sah berstatus E3 (0.70) sedangkan pengurasan tanpa cadangan /
    eksploitasi akses berstatus E4 (1.00) sesuai Persamaan (3.4).
    """
    if not tx_ok:
        return

    cid = env_state["cid"]
    sig = arm.get("sig", "")
    val = float(arm.get("val", 0.0))

    # Setoran ETH sah dari agen meningkatkan kewajiban ETH kontrak
    if val > 0.0:
        env_state["eth_liability"] += val

    # Penarikan simpanan/bid sah milik agen menurunkan kewajiban ETH sebesar saldo milik agen yang ditarik
    if ("B01" in cid and sig == "withdraw()") or ("M02" in cid and sig.startswith("withdrawBid")):
        cur_eth = client.get_eth_balance(caddr)
        env_state["eth_liability"] = min(float(env_state["eth_liability"]), cur_eth)

    # Setoran / staking token sah dari agen meningkatkan kewajiban cadangan token kontrak
    if "S01" in cid and sig.startswith("deposit(uint256)"):
        amt_tok = float(int(arm["args"][0])) / 1e18
        env_state["tok_liability"] += amt_tok
    elif "D01" in cid and sig.startswith("stake(uint256,uint256)"):
        amt_tok = float(int(arm["args"][0])) / 1e18
        # Jika agen belum memasang referrer, stake menambah cadangan penuh;
        # jika agen telah memasang referrer, bonus referal 10% tak berjaminan menguras cadangan!
        env_state["tok_liability"] += amt_tok
        # Majukan waktu agar lockPeriod 30 hari siap diuji pada putaran berikutnya
        rpc_call("evm_increaseTime", [2600000])
        rpc_call("evm_mine", [])
    elif "D02" in cid and sig.startswith("stake(uint256,uint256)"):
        amt_tok = float(int(arm["args"][0])) / 1e18
        env_state["tok_liability"] += amt_tok
        # Majukan waktu agar lockup tier siap diuji pada putaran berikutnya
        rpc_call("evm_increaseTime", [604800])
        rpc_call("evm_mine", [])
    elif "D02" in cid and sig.startswith("unlockPremature"):
        # Penarikan prematur hanya mengembalikan pokok simpanan (reward = 0), bukan eksploitasi cadangan
        custody_tok_addr = env_state["custody_token_addr"]
        cur_tok = client.call_uint256_ether_with_fallbacks(
            custody_tok_addr, ["balanceOf(address)(uint256)"], caddr
        )
        env_state["tok_liability"] = min(float(env_state["tok_liability"]), cur_tok)
    elif "M02" in cid and sig.startswith("placeBid(uint256)"):
        # Majukan waktu agar batas withdrawBid (86400 detik) tercapai
        rpc_call("evm_increaseTime", [90000])
        rpc_call("evm_mine", [])


def reanchor_post_vulnerability_state(
    spec: Dict[str, Any],
    caddr: str,
    env_state: Dict[str, Any],
    client: UnprivilegedAnvilClient,
    breakdown: BanditRewardBreakdown,
) -> None:
    """
    Menyelaraskan kembali baseline L_{t+1}^(C) setelah orakel E4 tercatat pada putaran t,
    agar putaran berikutnya mengevaluasi transisi invarian baru secara bersih (Persamaan 3.4).
    """
    cid = env_state["cid"]
    custody_tok_addr = env_state["custody_token_addr"]

    if breakdown.event_partition == "E4":
        cur_eth = client.get_eth_balance(caddr)
        if "B01" in cid and cur_eth < 0.01:
            # Pulihkan state storage & likuiditas dasar 4.98 ETH pada VulnerableVault setelah selfdestruct/drain
            # (Cancun EIP-6780 mengosongkan saldo ETH tanpa menghapus slot storage pasca-deployment)
            zero_hex = "0x" + ("0" * 64)
            for addr in (AGENT_ADDR, PEER_ADDR, DEPLOYER_ADDR):
                rpc_call("anvil_setStorageAt", [caddr, compute_solidity_mapping_slot(addr, 0), zero_hex])
            rpc_call("anvil_setStorageAt", [caddr, "0x1", zero_hex])
            send_tx(DEPLOYER_PK, caddr, "deposit()", [], 4.48)
            send_tx(PEER_PK, caddr, "deposit()", [], 0.50)
            env_state["peer_deposited_eth"] = 0.50
        elif "M02" in cid and not check_shuffle_room2_intact(caddr):
            # Kembalikan kepemilikan fundsWallet Room 2 ke DEPLOYER_ADDR setelah pembajakan tercatat
            send_tx(
                DEPLOYER_PK,
                caddr,
                "modifyRoom(uint256,bool,uint256,address)",
                ["2", "true", "5", DEPLOYER_ADDR],
            )
        elif "D03" in cid:
            # Perbarui root Merkle untuk pengujian putaran selanjutnya
            merkle_root = compute_single_node_merkle_root(AGENT_ADDR, 250 * 10**18)
            send_tx(DEPLOYER_PK, caddr, "setRoot(bytes32)", [merkle_root])

    vault_eth = client.get_eth_balance(caddr)
    escrow_eth = get_escrow_balance(caddr, client) if "M01" in cid else 0.0
    env_state["eth_liability"] = vault_eth + escrow_eth

    contract_tok_res = client.call_uint256_ether_with_fallbacks(
        custody_tok_addr, ["balanceOf(address)(uint256)"], caddr
    )
    env_state["tok_liability"] = contract_tok_res


def score_transition(
    arm_id: int,
    tx_succeeded: bool,
    prev_snap: OnChainContextSnapshot,
    next_snap: OnChainContextSnapshot,
) -> BanditRewardBreakdown:
    """
    Evaluasi imbalan berjenjang 4-komponen menggunakan VaultCustodyRewardOracle
    sesuai Persamaan (3.3), (3.4), (3.5), Tabel 3.1, dan Tabel 3.2:
        supp(r_t) = {0.00, 0.20, 0.50, 0.70, 1.00}.
    """
    return REWARD_ORACLE.compute_reward(
        prev_snap=prev_snap,
        next_snap=next_snap,
        tx_succeeded=tx_succeeded,
        arm_id=arm_id,
    )


def run_single_policy(
    policy_name: str,
    policy: Any,
    spec: Dict[str, Any],
    contract_bytecode: str,
    mock_token_bytecode: str,
    network_events: List[Dict[str, Any]],
    gas_prices: List[float],
    initial_budget: float,
    horizon_t: int,
    min_capital_threshold: float = MIN_CAPITAL_THRESHOLD_ETH,
) -> Dict[str, Any]:
    mock_token_addr = deploy_bytecode(
        mock_token_bytecode, "constructor(uint256)", ["1000000000000000000000000"]
    )
    caddr = deploy_bytecode(contract_bytecode, spec.get("constructor_sig"), spec.get("constructor_args"))
    if not caddr or not mock_token_addr:
        raise RuntimeError(f"Deploy failed for {spec['contract']}")

    client = UnprivilegedAnvilClient(rpc_url=RPC_URL)
    if spec["contract"] in COMPILED_ABIS:
        client.register_contract_abi(caddr, COMPILED_ABIS[spec["contract"]])

    env_state = setup_member_environment(spec, caddr, mock_token_addr, client)

    # Pastikan dompet EOA agen memiliki ETH yang cukup di Anvil untuk menandatangani transaksi,
    # sementara anggaran modal gas B_t dilacak secara presisi sesuai Persamaan (3.11) & (3.12).
    rpc_call("anvil_setBalance", [AGENT_ADDR, hex(int(max(10.0, initial_budget * 5.0) * 1e18))])
    remaining_budget = float(initial_budget)

    arms = spec["arms"]
    K = len(arms)
    pos_vals = [float(a.get("val", 0.0)) for a in arms if float(a.get("val", 0.0)) > 0.0]
    min_dep_req = min(pos_vals) if pos_vals else 0.05
    extractor = PublicContextExtractor(
        min_deposit_eth=min_dep_req,
        vault_cap_eth=20.0,
        agent_init_eth=initial_budget,
    )

    rewards: List[float] = []
    costs: List[float] = []
    decision_times_ms: List[float] = []
    round_logs: List[Dict[str, Any]] = []
    arm_counts = [0] * K
    event_counts: Dict[str, int] = {"E0": 0, "E1": 0, "E2": 0, "E3": 0, "E4": 0}
    success_count = 0
    sstore_count = 0
    waypoint_count = 0
    vuln_trigger_count = 0
    anomalies_found: List[str] = []
    recent_reverts: List[int] = []
    last_sstore_flag = False
    last_waypoint_flag = False
    termination_reason = "completed"

    for t in range(1, horizon_t + 1):
        # Algoritma 3.1 baris 7-9: Kondisi penghentian awal jika B_{t-1} <= B_min
        if remaining_budget <= min_capital_threshold:
            termination_reason = "bankruptcy"
            break

        current_gas_gwei = gas_prices[t - 1] if t - 1 < len(gas_prices) else 20.0
        rpc_call("anvil_setNextBlockBaseFeePerGas", [hex(int(current_gas_gwei * 1e9))])

        if t - 1 < len(network_events):
            execute_network_event(network_events[t - 1], caddr, mock_token_addr, env_state, client)

        # Moving average tingkat kegagalan dengan lebar jendela W = 8 (Tabel 3.1)
        rev_rate = float(np.mean(recent_reverts[-8:])) if recent_reverts else 0.0
        vec, prev_snap = build_context_vector(
            caddr=caddr,
            mock_token_addr=mock_token_addr,
            client=client,
            extractor=extractor,
            env_state=env_state,
            recent_revert_rate=rev_rate,
            agent_budget_eth=remaining_budget,
            sstore_executed=last_sstore_flag,
            dataflow_waypoint=last_waypoint_flag,
            min_deposit_requirement=min_dep_req,
        )

        # Pengukuran waktu komputasi seleksi keputusan (Metrik 5 - Sub-bab 3.6)
        t0_dec = time.perf_counter()
        sel = policy.select_arm(vec)
        t1_dec = time.perf_counter()

        arm_idx = sel.arm_id
        chosen_arm = arms[arm_idx]
        arm_counts[arm_idx] += 1

        target_for_trace = mock_token_addr if chosen_arm.get("target") == "MOCK_TOKEN" else caddr
        pre_storage = client.snapshot_contract_storage(target_for_trace)
        pre_agent_eoa_eth = client.get_eth_balance(AGENT_ADDR)

        tx_ok, tx_out, resolved_args = execute_member_arm(
            spec, chosen_arm, caddr, mock_token_addr, t, gas_price_gwei=current_gas_gwei
        )
        if tx_ok:
            success_count += 1
        recent_reverts.append(0 if tx_ok else 1)

        tx_hash = client.extract_tx_hash(tx_out) if tx_ok else None
        sstore_val, waypoint_val = client.trace_transaction_dataflow(
            tx_hash=tx_hash,
            contract_address=target_for_trace,
            pre_storage=pre_storage,
        )
        last_sstore_flag = bool(sstore_val >= 0.5)
        last_waypoint_flag = bool(waypoint_val >= 0.5)
        if last_sstore_flag:
            sstore_count += 1
        if last_waypoint_flag:
            waypoint_count += 1

        update_legitimate_liabilities_before_post_snapshot(
            spec, chosen_arm, tx_ok, caddr, env_state, client
        )

        post_agent_eoa_eth = client.get_eth_balance(AGENT_ADDR)
        eoa_eth_delta = post_agent_eoa_eth - pre_agent_eoa_eth

        _, next_snap = build_context_vector(
            caddr=caddr,
            mock_token_addr=mock_token_addr,
            client=client,
            extractor=extractor,
            env_state=env_state,
            recent_revert_rate=rev_rate,
            agent_budget_eth=remaining_budget + max(0.0, eoa_eth_delta),
            sstore_executed=last_sstore_flag,
            dataflow_waypoint=last_waypoint_flag,
            min_deposit_requirement=min_dep_req,
        )

        breakdown = score_transition(arm_idx, tx_ok, prev_snap, next_snap)
        rew = breakdown.total_reward
        anoms = breakdown.detected_anomalies
        ev_part = breakdown.event_partition
        event_counts[ev_part] = event_counts.get(ev_part, 0) + 1
        if ev_part == "E4":
            vuln_trigger_count += 1

        reanchor_post_vulnerability_state(spec, caddr, env_state, client, breakdown)

        # Persamaan (3.11): c_t(a_t) = clip((GasUsed_t(a_t) * g_t) + v_{a_t}, c_min, c_max)
        gas_used = parse_gas_used(tx_ok, tx_out, n_args=len(resolved_args))
        g_t_eth = float(current_gas_gwei) * 1e-9
        v_at = float(chosen_arm.get("val", 0.0))
        tx_cost = float(np.clip((gas_used * g_t_eth) + v_at, 0.001, 1.0))

        # Persamaan (3.12): B_t = max(0, B_{t-1} - c_t(a_t))
        remaining_budget = max(0.0, remaining_budget - tx_cost)

        t2_dec = time.perf_counter()
        try:
            policy.update(arm_idx, vec, rew, cost=tx_cost)
        except TypeError:
            policy.update(arm_idx, vec, rew)
        t3_dec = time.perf_counter()

        overhead_ms = ((t1_dec - t0_dec) + (t3_dec - t2_dec)) * 1000.0
        decision_times_ms.append(overhead_ms)

        rewards.append(rew)
        costs.append(tx_cost)
        if anoms:
            anomalies_found.extend(anoms)

        ev_name = network_events[t - 1]["type"] if (t - 1 < len(network_events)) else "none"
        dual_val = float(getattr(policy, "lambda_t", 0.0))

        round_log = {
            "round": t,
            "gas_price_gwei": round(current_gas_gwei, 2),
            "gas_used": gas_used,
            "arm_id": arm_idx,
            "arm_name": chosen_arm["name"],
            "arm_role": chosen_arm.get("role", ""),
            "tx_ok": tx_ok,
            "event_partition": ev_part,
            "sstore_executed": last_sstore_flag,
            "dataflow_waypoint": last_waypoint_flag,
            "reward": round(rew, 4),
            "cost_eth": round(tx_cost, 6),
            "gas_cost": round(tx_cost, 6),
            "cumulative_reward": round(float(np.sum(rewards)), 4),
            "balance_eth": round(remaining_budget, 6),
            "burned_eth": round(float(np.sum(costs)), 6),
            "dual_lambda": round(dual_val, 4),
            "decision_overhead_ms": round(overhead_ms, 4),
            "event_type": ev_name,
        }
        round_logs.append(round_log)

        # Algoritma 3.1 baris 23-24: Hentikan jika sisa anggaran B_t <= B_min
        if remaining_budget <= min_capital_threshold:
            termination_reason = "bankruptcy"
            break

    actual_rounds = len(rewards) if rewards else 0
    capital_burned = float(np.sum(costs)) if costs else 0.0
    cum_reward = float(np.sum(rewards)) if rewards else 0.0

    # Persamaan (3.23): Eff_gas = sum_{t=1}^tau r_t(a_t) / sum_{t=1}^tau c_t(a_t)
    eff_gas = round(cum_reward / max(1e-6, capital_burned), 4)

    # Tingkat Kelangsungan Hidup Anggaran (Sub-bab 3.6 Butir 4): SR = tau / T
    survival_rate = round(float(actual_rounds) / float(max(1, horizon_t)), 4)

    probs = np.array(arm_counts, dtype=float) / float(max(1, actual_rounds))
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0

    mean_overhead_ms = round(float(np.mean(decision_times_ms)), 4) if decision_times_ms else 0.0
    std_overhead_ms = round(float(np.std(decision_times_ms)), 4) if decision_times_ms else 0.0

    return {
        "policy": policy_name,
        "contract": spec["contract"],
        "rounds_survived": actual_rounds,
        "budget_survival_rate": survival_rate,
        "termination_reason": termination_reason,
        "bankruptcy": (termination_reason == "bankruptcy"),
        "initial_budget_eth": initial_budget,
        "final_capital_eth": round(remaining_budget, 4),
        "capital_burned_eth": round(capital_burned, 4),
        "gas_efficiency_ratio": eff_gas,
        "capital_efficiency_ratio": eff_gas,
        "cumulative_reward": round(cum_reward, 4),
        "mean_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
        "vuln_trigger_count": vuln_trigger_count,
        "sstore_count": sstore_count,
        "waypoint_count": waypoint_count,
        "event_counts": event_counts,
        "success_rate": round(float(success_count) / float(max(1, actual_rounds)), 4),
        "revert_count": actual_rounds - success_count,
        "arm_counts": arm_counts,
        "arm_entropy_bits": round(entropy, 4),
        "final_dual_lambda": round(float(getattr(policy, "lambda_t", 0.0)), 4),
        "mean_execution_overhead_ms": mean_overhead_ms,
        "std_execution_overhead_ms": std_overhead_ms,
        "anomalies": sorted(list(set(anomalies_found))),
        "round_logs": round_logs,
    }


def _student_t_crit_95(df: int) -> float:
    """Nilai kritis distribusi Student-t dua sisi (alpha = 0.05) untuk derajat bebas df."""
    table = {
        1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706,
        6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281,
        11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
        16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860,
        25: 2.0595, 30: 2.0423,
    }
    if df in table:
        return table[df]
    if df < 30:
        return 2.0423 + (30 - df) * 0.008
    return 1.95996 + (2.37226 / float(df)) + (2.8225 / float(df * df))


def _student_t_two_sided_pvalue(t_abs: float, df: int) -> float:
    """Menghitung p-value dua sisi distribusi Student-t secara numerik (aturan Simpson)."""
    if t_abs <= 0.0:
        return 1.0
    if df <= 0:
        return 1.0
    # Transformasi integral: x = t * sqrt(df), gunakan substitusi u = df / (df + t^2) pada Beta reguler
    # atau integrasi Simpson langsung dari 0 hingga t_abs pada kepadatan Student-t
    n_steps = 400
    v = float(df)
    log_norm = (
        math.lgamma((v + 1.0) / 2.0)
        - 0.5 * math.log(v * math.pi)
        - math.lgamma(v / 2.0)
    )
    upper = min(float(t_abs), 50.0)
    u = np.linspace(0.0, upper, n_steps + 1)
    pdf = np.exp(log_norm - 0.5 * (v + 1.0) * np.log1p((u * u) / v))
    weights = np.ones(n_steps + 1)
    weights[1:-1:2] = 4.0
    weights[2:-2:2] = 2.0
    integral = float((upper / (3.0 * n_steps)) * np.sum(weights * pdf))
    p_val = max(0.0, min(1.0, 2.0 * (0.5 - integral)))
    return p_val


def _paired_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Uji hipotesis berpasangan Student paired t-test."""
    diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    n = len(diff)
    if n < 2:
        return 0.0, 1.0
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    if std_d <= 1e-12:
        return (0.0, 1.0) if abs(mean_d) <= 1e-12 else (float(np.sign(mean_d) * 999.0), 0.0)
    t_stat = mean_d / (std_d / math.sqrt(n))
    p_val = _student_t_two_sided_pvalue(abs(t_stat), df=n - 1)
    return t_stat, p_val


def _wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Uji non-parametrik berpasangan Wilcoxon Signed-Rank Test (eksak untuk m <= 15, normal untuk m > 15)."""
    diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    nz = diff[np.abs(diff) > 1e-12]
    m = len(nz)
    if m < 2:
        return 0.0, 1.0

    abs_nz = np.abs(nz)
    order = np.argsort(abs_nz)
    ranks = np.empty(m, dtype=np.float64)
    tie_counts: List[int] = []

    i = 0
    while i < m:
        j = i
        while j + 1 < m and abs(abs_nz[order[j + 1]] - abs_nz[order[i]]) <= 1e-12:
            j += 1
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        if j > i:
            tie_counts.append(j - i + 1)
        i = j + 1

    w_pos = float(np.sum(ranks[nz > 0]))
    w_neg = float(np.sum(ranks[nz < 0]))
    w_stat = min(w_pos, w_neg)
    total_rank = float(np.sum(ranks))

    if m <= 15:
        # Enumerasi kombinatorial eksak 2^m
        extreme_count = 0
        total_perms = 1 << m
        for mask in range(total_perms):
            s_pos = 0.0
            for bit_idx in range(m):
                if (mask >> bit_idx) & 1:
                    s_pos += ranks[bit_idx]
            s_min = min(s_pos, total_rank - s_pos)
            if s_min <= w_stat + 1e-9:
                extreme_count += 1
        p_val = min(1.0, float(extreme_count) / float(total_perms))
        return w_stat, p_val

    mu_w = m * (m + 1.0) / 4.0
    tie_corr = sum((t * t * t - t) for t in tie_counts) / 2.0
    var_w = (m * (m + 1.0) * (2.0 * m + 1.0) - tie_corr) / 24.0
    if var_w <= 1e-12:
        return w_stat, 1.0
    z = ( abs(w_stat - mu_w) - 0.5 ) / math.sqrt(var_w)
    p_val = max(0.0, min(1.0, math.erfc(max(0.0, z) / math.sqrt(2.0))))
    return w_stat, p_val


def compute_statistical_summary(all_experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Menghitung agregasi statistik (mu +/- sigma, 95% CI) dan uji signifikansi berpasangan
    (paired t-test & Wilcoxon Signed-Rank Test pada alpha_stat = 0.05) sesuai Sub-bab 3.6 Butir 5.
    """
    policy_names = ["UniformRandom", "LinUCB", "D-LinUCB", "LinCBwK", "D-LinCBwK"]
    metrics = [
        "cumulative_dynamic_regret",
        "cumulative_reward",
        "vuln_trigger_count",
        "sstore_count",
        "waypoint_count",
        "gas_efficiency_ratio",
        "budget_survival_rate",
        "mean_execution_overhead_ms",
    ]

    collected: Dict[str, Dict[str, List[float]]] = {
        p: {m: [] for m in metrics} for p in policy_names
    }

    for exp in all_experiments:
        res_map = exp.get("results", {})
        for p in policy_names:
            if p in res_map:
                for m in metrics:
                    collected[p][m].append(float(res_map[p].get(m, 0.0)))

    summary_by_policy: Dict[str, Dict[str, Any]] = {}
    for p in policy_names:
        summary_by_policy[p] = {}
        for m in metrics:
            arr = np.array(collected[p][m], dtype=np.float64)
            n = len(arr)
            if n == 0:
                continue
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            if n > 1 and std_val > 0.0:
                t_crit = _student_t_crit_95(n - 1)
                margin = t_crit * (std_val / math.sqrt(n))
            else:
                margin = 0.0
            summary_by_policy[p][m] = {
                "n": n,
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "ci95_low": round(mean_val - margin, 4),
                "ci95_high": round(mean_val + margin, 4),
                "formatted": f"{mean_val:.4f} +/- {std_val:.4f}",
            }

    # Uji signifikansi berpasangan: D-LinCBwK vs setiap baseline pada alpha_stat = 0.05
    paired_tests: Dict[str, Dict[str, Any]] = {}
    proposed = "D-LinCBwK"
    for base_p in ["UniformRandom", "LinUCB", "D-LinUCB", "LinCBwK"]:
        paired_tests[f"{proposed}_vs_{base_p}"] = {}
        for m in metrics:
            x = np.array(collected[proposed][m], dtype=np.float64)
            y = np.array(collected[base_p][m], dtype=np.float64)
            n = min(len(x), len(y))
            if n < 2 or np.allclose(x[:n], y[:n]):
                paired_tests[f"{proposed}_vs_{base_p}"][m] = {
                    "n_pairs": n,
                    "mean_diff": round(float(np.mean(x[:n] - y[:n])) if n > 0 else 0.0, 4),
                    "t_stat": 0.0,
                    "p_value_ttest": 1.0,
                    "wilcoxon_stat": 0.0,
                    "p_value_wilcoxon": 1.0,
                    "significant_at_005": False,
                }
                continue

            diff = x[:n] - y[:n]
            t_stat, p_ttest = _paired_ttest(x[:n], y[:n])
            w_stat, p_wilcox = _wilcoxon_signed_rank(x[:n], y[:n])

            paired_tests[f"{proposed}_vs_{base_p}"][m] = {
                "n_pairs": n,
                "mean_diff": round(float(np.mean(diff)), 4),
                "t_stat": round(t_stat, 4),
                "p_value_ttest": round(p_ttest, 6),
                "wilcoxon_stat": round(w_stat, 4),
                "p_value_wilcoxon": round(p_wilcox, 6),
                "significant_at_005": bool((p_ttest < 0.05) or (p_wilcox < 0.05)),
            }

    return {
        "alpha_stat": 0.05,
        "policy_metrics": summary_by_policy,
        "paired_hypothesis_tests": paired_tests,
    }


def resolve_contract_path(c_spec: Dict[str, Any]) -> Path:
    if c_spec.get("split") == "benchmark" or c_spec["file"] == "VulnerableVault.sol":
        return VULNERABLE_VAULT_SOL
    return DATASET_DIR / c_spec["split"] / c_spec["file"]


def run_experiment_matrix(
    tier: str = "all",
    regime: str = "all",
    budget: str = "all",
    horizon: int = DEFAULT_HORIZON_T,
    contracts_limit: Optional[int] = None,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    seeds: int = 1,
) -> Dict[str, Any]:
    ensure_anvil_running()

    tiers = ["benchmark", "shallow", "medium", "deep"] if tier == "all" else [tier]
    regimes = ["stable", "volatile_spikes"] if regime == "all" else [regime]
    budgets = ["tight", "moderate", "loose"] if budget == "all" else [budget]

    mock_token_bin = compile_contract(MOCK_TOKEN_SOL, "MockToken")
    if not mock_token_bin:
        raise RuntimeError("Gagal kompilasi MockToken.sol")

    all_experiments: List[Dict[str, Any]] = []
    t_start = time.time()

    print("=" * 104)
    print(
        f"🚀 MEMULAI EKSPERIMEN SKRIPSI FAKTORIAL | T = {horizon} Putaran | "
        f"Alpha = {alpha} | Gamma = {gamma} | Seeds M = {seeds}"
    )
    print(f"   Kondisi 1 (Tiers): {tiers}")
    print(f"   Kondisi 2 (Regimes): {regimes}")
    print(f"   Kondisi 3 (Budgets): {budgets}")
    print("=" * 104)

    for t_name in tiers:
        contracts = get_contracts_by_tier(t_name)
        if contracts_limit:
            contracts = contracts[:contracts_limit]

        for c_spec in contracts:
            cname = c_spec["contract"]
            c_idx = int(c_spec.get("contract_idx", 1))
            fpath = resolve_contract_path(c_spec)
            print(f"\n[+] Kompilasi Kontrak: {cname} (Tier: {t_name.upper()} | {c_spec['file']})")
            c_bin = compile_contract(fpath, cname)
            if not c_bin:
                print(f"    [!] Gagal kompilasi {cname}; dilewati.")
                continue

            for reg_name in regimes:
                for b_name in budgets:
                    b_conf = BUDGET_TIERS[b_name]
                    init_b = b_conf["budget_eth"]

                    for seed_idx in range(seeds):
                        crn_seed = 2026 + seed_idx * 101
                        gas_series = generate_gas_series(
                            reg_name,
                            rounds=horizon,
                            seed=30300 + c_idx * 41 + seed_idx * 997,
                        )
                        net_events = pregenerate_network_events(
                            c_idx, rounds=horizon, seed_offset=seed_idx
                        )

                        seed_label = f" | Seed #{seed_idx + 1}/{seeds} ({crn_seed})" if seeds > 1 else ""
                        print(
                            f"\n---> MENGEKSEKUSI: {cname} | Rezim: {reg_name.upper()} | "
                            f"Anggaran: {b_name.upper()} ({init_b} ETH){seed_label}"
                        )

                        K = len(c_spec["arms"])

                        # 1. UniformRandom (Forge Fuzz Baseline)
                        p_random = UniformRandomPolicy(
                            n_arms=K,
                            context_dim=CONTEXT_DIM,
                            initial_budget=init_b,
                            total_horizon=horizon,
                            seed=crn_seed,
                            name="UniformRandom",
                        )
                        # 2. LinUCB (Stationary Unconstrained: gamma = 1.0, lambda_t = 0)
                        p_linucb = DisjointLinUCB(
                            n_arms=K,
                            context_dim=CONTEXT_DIM,
                            alpha=alpha,
                            l2_lambda=L2_LAMBDA,
                            seed=crn_seed,
                        )
                        # 3. D-LinUCB (Discounted Unconstrained: gamma in (0,1), lambda_t = 0)
                        p_dlinucb = DiscountedLinUCB(
                            n_arms=K,
                            context_dim=CONTEXT_DIM,
                            gamma=gamma,
                            alpha=alpha,
                            l2_lambda=L2_LAMBDA,
                            use_theoretical_beta=False,
                            decay_all_arms=False,
                            seed=crn_seed,
                        )
                        # 4. LinCBwK (Stationary Knapsack: gamma = 1.0, dynamic lambda_t)
                        p_lincbwk = StationaryLinCBwK(
                            n_arms=K,
                            context_dim=CONTEXT_DIM,
                            initial_budget=init_b,
                            total_horizon=horizon,
                            alpha=alpha,
                            l2_lambda=L2_LAMBDA,
                            initial_lambda=INITIAL_LAMBDA,
                            dual_update_mode="observed",
                            seed=crn_seed,
                            name="LinCBwK",
                        )
                        # 5. D-LinCBwK (Discounted Knapsack - Proposed: gamma in (0,1), dynamic lambda_t)
                        p_dlincbwk = DiscountedLinCBwK(
                            n_arms=K,
                            context_dim=CONTEXT_DIM,
                            gamma=gamma,
                            initial_budget=init_b,
                            total_horizon=horizon,
                            alpha=alpha,
                            l2_lambda=L2_LAMBDA,
                            initial_lambda=INITIAL_LAMBDA,
                            dual_update_mode="observed",
                            decay_all_arms=False,
                            seed=crn_seed,
                            name="D-LinCBwK",
                        )

                        policies = [
                            ("UniformRandom", p_random),
                            ("LinUCB", p_linucb),
                            ("D-LinUCB", p_dlinucb),
                            ("LinCBwK", p_lincbwk),
                            ("D-LinCBwK", p_dlincbwk),
                        ]

                        cell_results: Dict[str, Any] = {}
                        policies_round_logs: Dict[str, List[Dict[str, Any]]] = {}

                        for p_tag, p_inst in policies:
                            res = run_single_policy(
                                policy_name=p_tag,
                                policy=p_inst,
                                spec=c_spec,
                                contract_bytecode=c_bin,
                                mock_token_bytecode=mock_token_bin,
                                network_events=net_events,
                                gas_prices=gas_series,
                                initial_budget=init_b,
                                horizon_t=horizon,
                            )
                            cell_results[p_tag] = res
                            policies_round_logs[p_tag] = res["round_logs"]

                        # Evaluasi Metrik 1: Cumulative Dynamic Regret D-Regret(T) (Persamaan 3.22)
                        regret_summaries = compare_policies_dynamic_regret(
                            policies_round_logs=policies_round_logs,
                            context_dim=CONTEXT_DIM,
                            n_arms=K,
                            initial_budget=init_b,
                            oracle_reward_ceiling=1.00,
                            horizon_t=horizon,
                        )
                        for p_tag, reg_sum in regret_summaries.items():
                            cell_results[p_tag]["cumulative_dynamic_regret"] = round(
                                reg_sum.cumulative_dynamic_regret, 4
                            )
                            cell_results[p_tag]["mean_dynamic_regret"] = round(
                                reg_sum.mean_dynamic_regret, 4
                            )
                            cell_results[p_tag]["state_switches_S"] = reg_sum.state_switches
                            cell_results[p_tag]["dynamic_regret_trajectory"] = [
                                round(step.cumulative_regret, 4) for step in reg_sum.trajectory
                            ]
                            res = cell_results[p_tag]
                            print(
                                f"     * {p_tag:<14}: SR={res['budget_survival_rate']*100:5.1f}% "
                                f"({res['rounds_survived']:2d}/{horizon}) | "
                                f"Rew={res['cumulative_reward']:5.2f} | "
                                f"D-Regret={res['cumulative_dynamic_regret']:5.2f} | "
                                f"Vuln(E4)={res['vuln_trigger_count']:2d} | "
                                f"Eff_gas={res['gas_efficiency_ratio']:6.2f} | "
                                f"Overhead={res['mean_execution_overhead_ms']:.3f}ms"
                            )

                        experiment_record = {
                            "contract_id": c_spec["id"],
                            "contract_name": cname,
                            "tier": t_name,
                            "gas_regime": reg_name,
                            "budget_tier": b_name,
                            "seed_idx": seed_idx,
                            "crn_seed": crn_seed,
                            "initial_budget_eth": init_b,
                            "horizon_t": horizon,
                            "results": cell_results,
                        }
                        all_experiments.append(experiment_record)

    t_elapsed = time.time() - t_start
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = TELEMETRY_DIR / f"skripsi_experiments_{timestamp_str}.json"
    latest_file = TELEMETRY_DIR / "skripsi_experiments_latest.json"

    stat_summary = compute_statistical_summary(all_experiments)

    payload = {
        "timestamp": timestamp_str,
        "elapsed_seconds": round(t_elapsed, 2),
        "hyperparameters": {
            "alpha": alpha,
            "gamma": gamma,
            "l2_lambda": L2_LAMBDA,
            "horizon_t": horizon,
            "seeds_m": seeds,
            "reward_weights": [0.20, 0.30, 0.20, 1.00],
            "reward_support": [0.00, 0.20, 0.50, 0.70, 1.00],
        },
        "total_experiments": len(all_experiments),
        "statistical_summary": stat_summary,
        "experiments": all_experiments,
    }

    out_file.write_text(json.dumps(payload, indent=2))
    latest_file.write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 104)
    print(f"🏁 EKSPERIMEN SKRIPSI SELESAI DALAM {t_elapsed:.2f} DETIK")
    print(f"   Total Evaluasi Grid: {len(all_experiments)} Matriks Kasus")
    print(f"   Telemetri Disimpan ke: {out_file}")
    print("=" * 104)

    return payload


def main():
    parser = argparse.ArgumentParser(description="Runner Eksperimen Skripsi Multi-Kondisi (Anvil EVM)")
    parser.add_argument(
        "--tier",
        choices=["all", "benchmark", "shallow", "medium", "deep"],
        default="all",
        help="Kondisi 1: Kelompok/Kedalaman State Kontrak",
    )
    parser.add_argument(
        "--regime",
        choices=["all", "stable", "volatile_spikes"],
        default="all",
        help="Kondisi 2: Dinamika Gas",
    )
    parser.add_argument(
        "--budget",
        choices=["all", "tight", "moderate", "loose"],
        default="all",
        help="Kondisi 3: Modal Gas B",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON_T,
        help=f"Horizon T (default: {DEFAULT_HORIZON_T})",
    )
    parser.add_argument(
        "--contracts-limit",
        type=int,
        default=None,
        help="Batas jumlah kontrak per tier untuk quick run",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="Hyperparameter Alpha (default: 0.15)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=GAMMA,
        help="Hyperparameter Gamma (default: 0.82)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Jumlah replikasi independen M berbasis CRN (Sub-bab 3.5 Tahap 3)",
    )

    args = parser.parse_args()
    run_experiment_matrix(
        tier=args.tier,
        regime=args.regime,
        budget=args.budget,
        horizon=args.horizon,
        contracts_limit=args.contracts_limit,
        alpha=args.alpha,
        gamma=args.gamma,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
