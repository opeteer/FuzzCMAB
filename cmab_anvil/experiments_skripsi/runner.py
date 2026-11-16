"""
Runner Utama Eksperimen Skripsi: Pengujian Faktorial Multi-Kondisi pada Anvil.

Mengeksekusi matriks pengujian:
  - Kondisi 1: Kedalaman State (Shallow, Medium, Deep)
  - Kondisi 2: Dinamika Harga Gas (Rezim 1 Stabil vs Rezim 2 Sinusoidal Burst + Spikes)
  - Kondisi 3: Anggaran Modal Gas (Tight, Moderate, Loose)
Membandingkan 5 kebijakan:
  1. UniformRandom (Forge Fuzz Baseline)
  2. LinUCB (Stationary Unconstrained)
  3. D-LinUCB (Discounted Unconstrained)
  4. LinCBwK (Stationary Knapsack)
  5. D-LinCBwK (Discounted Knapsack - Proposed)
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
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
from cmab_anvil.common.anvil_rpc import UnprivilegedAnvilClient
from cmab_anvil.Agen_Vault_Custody.context import (
    PublicContextExtractor,
    OnChainContextSnapshot,
    CONTEXT_DIM,
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
from cmab_anvil.experiments_skripsi.gas_regimes import generate_gas_series, get_gas_regime_metadata
from cmab_anvil.experiments_skripsi.contract_catalog import (
    CATALOG_CONTRACTS,
    TIER_METADATA,
    get_contracts_by_tier,
)


COMPILED_ABIS: Dict[str, List[Dict[str, Any]]] = {}


def ensure_anvil_running() -> None:
    p = subprocess.run([CAST_BIN, "block-number", "--rpc-url", RPC_URL], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
    res = subprocess.run([SOLC_BIN, "--combined-json", "abi,bin", str(fpath)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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


def deploy_bytecode(bytecode: str, constructor_sig: Optional[str] = None, constructor_args: Optional[List[str]] = None) -> Optional[str]:
    create_hex = "0x" + bytecode
    if constructor_sig and constructor_args:
        ok, encoded, _ = run_cast(["abi-encode", constructor_sig] + constructor_args)
        if ok and encoded.startswith("0x"):
            create_hex += encoded[2:]
    rpc_call("anvil_setNextBlockBaseFeePerGas", ["0x3b9aca00"])  # 1 Gwei
    ok, out, _ = run_cast(["send", "--private-key", DEPLOYER_PK, "--gas-price", "5000000000", "--create", create_hex])
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


def setup_member_environment(spec: Dict[str, Any], caddr: str, mock_token_addr: str) -> None:
    cid = spec["id"]
    MAX_UINT256 = "115792089237316195423570985008687907853269984665640564039457584007913129639935"

    send_tx(DEPLOYER_PK, mock_token_addr, "mint(address,uint256)", [AGENT_ADDR, "100000000000000000000000"])
    send_tx(DEPLOYER_PK, mock_token_addr, "mint(address,uint256)", [PEER_ADDR, "100000000000000000000000"])
    send_tx(DEPLOYER_PK, mock_token_addr, "mint(address,uint256)", [caddr, "100000000000000000000000"])
    send_tx(AGENT_PK, mock_token_addr, "approve(address,uint256)", [caddr, MAX_UINT256])

    if "S01" in cid:
        send_tx(DEPLOYER_PK, caddr, "transfer(address,uint256)", [AGENT_ADDR, "50000000000000000000"])
        send_tx(DEPLOYER_PK, caddr, "", [], 3.0)
    elif "M01" in cid:
        send_tx(DEPLOYER_PK, caddr, "setCost(uint256)", ["10000000000000000"])
        send_tx(PEER_PK, caddr, "mint(string)", ["STATIC_DUPLICATE"], 0.02)
    elif "D01" in cid:
        send_tx(DEPLOYER_PK, caddr, "transfer(address,uint256)", [AGENT_ADDR, "5000000000000000000000"])
        send_tx(AGENT_PK, caddr, "approve(address,uint256)", [caddr, MAX_UINT256])
    elif "D02" in cid:
        send_tx(DEPLOYER_PK, caddr, "initialize(address,address)", [mock_token_addr, DEPLOYER_ADDR])
        send_tx(DEPLOYER_PK, caddr, "setReward(uint256,uint256)", ["1", "500"])
    elif "D03" in cid:
        send_tx(DEPLOYER_PK, mock_token_addr, "mint(address,uint256)", [caddr, "500000000000000000000000"])


def pregenerate_network_events(contract_idx: int, rounds: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed=202600 + contract_idx * 37)
    events = []
    for t in range(1, rounds + 1):
        ev_type = rng.choice(
            ["peer_liquidity", "peer_withdrawal", "block_jitter", "congestion_poke", "none"],
            p=[0.25, 0.25, 0.20, 0.15, 0.15],
        )
        eth_delta = float(np.round(rng.uniform(0.10, 0.50), 3))
        time_delta = int(rng.integers(30, 900))
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
    peer_state: Dict[str, float],
    client: UnprivilegedAnvilClient,
) -> None:
    ev_type = ev["type"]
    if ev_type == "peer_liquidity":
        send_tx(PEER_PK, caddr, "", [], ev["eth_delta"])
        peer_state["deposited_eth"] = peer_state.get("deposited_eth", 0.0) + ev["eth_delta"]
    elif ev_type == "peer_withdrawal":
        avail = peer_state.get("deposited_eth", 0.0)
        c_bal = client.get_eth_balance(caddr)
        max_w = min(avail, c_bal)
        if max_w > 0.01:
            w_amt = float(np.round(min(max_w, ev["eth_delta"]), 4))
            peer_state["deposited_eth"] -= w_amt
            rpc_call("anvil_impersonateAccount", [caddr])
            send_tx(DEPLOYER_PK, caddr, "", [], 0.02)
            run_cast(["send", "--from", caddr, "--unlocked", "--value", f"{w_amt:.6f}ether", PEER_ADDR])
            rpc_call("anvil_stopImpersonatingAccount", [caddr])
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
    recent_revert_rate: float,
    round_idx: int,
    sstore_executed: bool = False,
    dataflow_waypoint: bool = False,
    min_deposit_requirement: float = 0.05,
) -> Tuple[np.ndarray, OnChainContextSnapshot]:
    block = client.get_block_number()
    vault_eth = client.get_eth_balance(caddr)
    agent_eth = client.get_eth_balance(AGENT_ADDR)
    code_alive = client.get_code_alive(caddr)

    agent_bal = client.query_agent_stake(caddr, AGENT_ADDR, fallback_token_address=mock_token_addr)
    total_dep = client.query_total_deposited(caddr, vault_eth_hint=vault_eth)
    token_bal = client.call_uint256_ether_with_fallbacks(mock_token_addr, ["balanceOf(address)(uint256)"], AGENT_ADDR)

    is_solvent = (vault_eth >= total_dep) if total_dep > 0 else True
    solv_ok, solv_val = client.call_bool(caddr, "check_solvency()(bool)")
    if solv_ok:
        is_solvent = solv_val

    lock_ok, locked_val = client.call_bool(caddr, "isLocked()(bool)")
    is_locked = locked_val if lock_ok else False

    snap = OnChainContextSnapshot(
        block_number=block,
        vault_eth_balance=vault_eth,
        vault_total_deposited=total_dep,
        agent_vault_balance=agent_bal,
        agent_eoa_eth_balance=agent_eth,
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
) -> Tuple[bool, str]:
    target = mock_token_addr if arm.get("target") == "MOCK_TOKEN" else caddr
    sig = arm["sig"]
    val = arm["val"]

    resolved_args = []
    for a in arm["args"]:
        if a == "DYNAMIC_URI":
            resolved_args.append(f"member_nft_{round_t}")
        elif a == "DYNAMIC_URI_LOW":
            resolved_args.append(f"member_low_{round_t}")
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
    return ok, out


def score_transition(
    spec: Dict[str, Any],
    arm: Dict[str, Any],
    tx_succeeded: bool,
    prev_snap: OnChainContextSnapshot,
    next_snap: OnChainContextSnapshot,
) -> Tuple[float, List[str]]:
    """
    Clean, arm-agnostic 4-component reward evaluation:
        r_t = 0.0 if REVERT
        r_t = min(1.0, R_exec + R_state + R_econ + R_vuln) if SUCCESS

    Zero hardcoded arm roles or function names. Evaluated strictly from on-chain EVM diffs.
    """
    anomalies: List[str] = []

    if not tx_succeeded:
        return 0.0, anomalies

    # 1. R_exec (+0.20): Valid on-chain execution
    r_exec = 0.20

    # 2. R_state (+0.30): Persistent storage state mutation (SSTORE)
    has_state_mutation = (
        getattr(next_snap, "sstore_executed", False)
        or abs(next_snap.agent_vault_balance - prev_snap.agent_vault_balance) > 1e-6
        or abs(next_snap.vault_total_deposited - prev_snap.vault_total_deposited) > 1e-6
    )
    r_state = 0.30 if has_state_mutation else 0.0

    # 3. R_econ (+0.20): Economic asset gain (ETH extracted or token acquired)
    eth_gain = next_snap.agent_eoa_eth_balance - prev_snap.agent_eoa_eth_balance
    tok_gain = next_snap.agent_token_balance - prev_snap.agent_token_balance
    gained_assets = (eth_gain > 0.001) or (tok_gain > 0.001)
    r_econ = 0.20 if gained_assets else 0.0

    if eth_gain > 0.001:
        anomalies.append(f"ETH_EXTRACTION (+{eth_gain:.3f} ETH)")
    if tok_gain > 0.001:
        anomalies.append(f"TOKEN_ACQUISITION (+{tok_gain:.0f} Tokens)")

    # 4. R_vuln (+0.30): Security invariant breach / exploit detection
    solvency_breached = prev_snap.vault_is_solvent and not next_snap.vault_is_solvent
    code_destroyed = prev_snap.vault_code_alive and not next_snap.vault_code_alive
    r_vuln = 0.30 if (solvency_breached or code_destroyed) else 0.0

    if solvency_breached:
        anomalies.append("SOLVENCY_INVARIANT_BREACH (vault_balance < totalDeposited)")
    if code_destroyed:
        anomalies.append("CONTRACT_SELFDESTRUCTED (code destroyed)")

    final_reward = round(min(1.0, r_exec + r_state + r_econ + r_vuln), 2)
    return final_reward, anomalies


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
    mock_token_addr = deploy_bytecode(mock_token_bytecode, "constructor(uint256)", ["1000000000000000000000000"])
    caddr = deploy_bytecode(contract_bytecode, spec.get("constructor_sig"), spec.get("constructor_args"))
    if not caddr or not mock_token_addr:
        raise RuntimeError(f"Deploy failed for {spec['contract']}")

    setup_member_environment(spec, caddr, mock_token_addr)
    send_tx(PEER_PK, caddr, "", [], 0.50)
    peer_state: Dict[str, float] = {"deposited_eth": 0.50}

    client = UnprivilegedAnvilClient(rpc_url=RPC_URL)
    if spec["contract"] in COMPILED_ABIS:
        client.register_contract_abi(caddr, COMPILED_ABIS[spec["contract"]])

    rpc_call("anvil_setBalance", [AGENT_ADDR, hex(int(initial_budget * 1e18))])

    arms = spec["arms"]
    K = len(arms)
    pos_vals = [float(a.get("val", 0.0)) for a in arms if float(a.get("val", 0.0)) > 0.0]
    min_dep_req = min(pos_vals) if pos_vals else 0.05
    extractor = PublicContextExtractor(min_deposit_eth=min_dep_req, vault_cap_eth=20.0, agent_init_eth=initial_budget)

    rewards: List[float] = []
    round_logs: List[Dict[str, Any]] = []
    arm_counts = [0] * K
    success_count = 0
    anomalies_found: List[str] = []
    recent_reverts: List[int] = []
    last_sstore_flag = False
    last_waypoint_flag = False
    termination_reason = "completed"

    for t in range(1, horizon_t + 1):
        current_bal = client.get_eth_balance(AGENT_ADDR)
        if current_bal <= min_capital_threshold:
            termination_reason = "bankruptcy"
            break

        current_gas_gwei = gas_prices[t - 1] if t - 1 < len(gas_prices) else 20.0
        rpc_call("anvil_setNextBlockBaseFeePerGas", [hex(int(current_gas_gwei * 1e9))])

        if t - 1 < len(network_events):
            execute_network_event(network_events[t - 1], caddr, mock_token_addr, peer_state, client)

        rev_rate = float(np.mean(recent_reverts[-8:])) if recent_reverts else 0.0
        vec, prev_snap = build_context_vector(
            caddr,
            mock_token_addr,
            client,
            extractor,
            rev_rate,
            t,
            sstore_executed=last_sstore_flag,
            dataflow_waypoint=last_waypoint_flag,
            min_deposit_requirement=min_dep_req,
        )

        sel = policy.select_arm(vec)
        arm_idx = sel.arm_id
        chosen_arm = arms[arm_idx]
        arm_counts[arm_idx] += 1

        target_for_trace = mock_token_addr if chosen_arm.get("target") == "MOCK_TOKEN" else caddr
        pre_storage = client.snapshot_contract_storage(target_for_trace)

        tx_ok, tx_out = execute_member_arm(spec, chosen_arm, caddr, mock_token_addr, t, gas_price_gwei=current_gas_gwei)
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

        _, next_snap = build_context_vector(
            caddr,
            mock_token_addr,
            client,
            extractor,
            rev_rate,
            t,
            sstore_executed=last_sstore_flag,
            dataflow_waypoint=last_waypoint_flag,
            min_deposit_requirement=min_dep_req,
        )
        rew, anoms = score_transition(spec, chosen_arm, tx_ok, prev_snap, next_snap)

        post_bal = client.get_eth_balance(AGENT_ADDR)
        tx_cost = max(0.0005, current_bal - post_bal)

        try:
            policy.update(arm_idx, vec, rew, cost=tx_cost)
        except TypeError:
            policy.update(arm_idx, vec, rew)

        rewards.append(rew)
        if anoms:
            anomalies_found.extend(anoms)

        ev_name = network_events[t - 1]["type"] if (t - 1 < len(network_events)) else "none"
        dual_val = getattr(policy, "lambda_t", 0.0)

        round_log = {
            "round": t,
            "gas_price_gwei": round(current_gas_gwei, 2),
            "arm_id": arm_idx,
            "arm_name": chosen_arm["name"],
            "arm_role": chosen_arm.get("role", ""),
            "tx_ok": tx_ok,
            "reward": round(rew, 4),
            "cost_eth": round(tx_cost, 6),
            "cumulative_reward": round(float(np.sum(rewards)), 4),
            "balance_eth": round(post_bal, 6),
            "burned_eth": round(max(0.0, initial_budget - post_bal), 6),
            "dual_lambda": round(dual_val, 4),
            "event_type": ev_name,
        }
        round_logs.append(round_log)

        if hasattr(policy, "check_stopping_condition"):
            stopped, stop_reason = policy.check_stopping_condition(vec)
            if stopped:
                termination_reason = stop_reason
                break

        if post_bal <= min_capital_threshold:
            termination_reason = "bankruptcy"
            break

    actual_rounds = len(rewards) if rewards else 1
    final_capital = client.get_eth_balance(AGENT_ADDR)
    capital_burned = max(0.0, initial_budget - final_capital)
    cum_reward = float(np.sum(rewards)) if rewards else 0.0
    cap_efficiency = round(cum_reward / max(0.001, capital_burned), 3)

    probs = np.array(arm_counts, dtype=float) / float(actual_rounds)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0

    return {
        "policy": policy_name,
        "contract": spec["contract"],
        "rounds_survived": actual_rounds,
        "termination_reason": termination_reason,
        "bankruptcy": (termination_reason == "bankruptcy"),
        "initial_budget_eth": initial_budget,
        "final_capital_eth": round(final_capital, 4),
        "capital_burned_eth": round(capital_burned, 4),
        "capital_efficiency_ratio": cap_efficiency,
        "cumulative_reward": round(cum_reward, 3),
        "mean_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
        "success_rate": round(float(success_count) / float(actual_rounds), 4),
        "revert_count": actual_rounds - success_count,
        "arm_counts": arm_counts,
        "arm_entropy_bits": round(entropy, 4),
        "final_dual_lambda": round(getattr(policy, "lambda_t", 0.0), 4),
        "anomalies": sorted(list(set(anomalies_found))),
        "round_logs": round_logs,
    }


def run_experiment_matrix(
    tier: str = "all",
    regime: str = "all",
    budget: str = "all",
    horizon: int = DEFAULT_HORIZON_T,
    contracts_limit: Optional[int] = None,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
) -> Dict[str, Any]:
    ensure_anvil_running()

    tiers = ["shallow", "medium", "deep"] if tier == "all" else [tier]
    regimes = ["stable", "volatile_spikes"] if regime == "all" else [regime]
    budgets = ["tight", "moderate", "loose"] if budget == "all" else [budget]

    mock_token_bin = compile_contract(MOCK_TOKEN_SOL, "MockToken")
    if not mock_token_bin:
        raise RuntimeError("Gagal kompilasi MockToken.sol")

    all_experiments: List[Dict[str, Any]] = []
    t_start = time.time()

    print("=" * 96)
    print(f"🚀 MEMULAI EKSPERIMEN SKRIPSI FAKTORIAL | T = {horizon} Putaran | Alpha = {alpha} | Gamma = {gamma}")
    print(f"   Kondisi 1 (Tiers): {tiers}")
    print(f"   Kondisi 2 (Regimes): {regimes}")
    print(f"   Kondisi 3 (Budgets): {budgets}")
    print("=" * 96)

    for t_idx, t_name in enumerate(tiers, 1):
        contracts = get_contracts_by_tier(t_name)
        if contracts_limit:
            contracts = contracts[:contracts_limit]

        for c_spec in contracts:
            cname = c_spec["contract"]
            fpath = DATASET_DIR / c_spec["split"] / c_spec["file"]
            print(f"\n[+] Kompilasi Kontrak: {cname} (Tier: {t_name.upper()} | {c_spec['file']})")
            c_bin = compile_contract(fpath, cname)
            if not c_bin:
                print(f"    [!] Gagal kompilasi {cname}; dilewati.")
                continue

            for reg_name in regimes:
                gas_series = generate_gas_series(reg_name, rounds=horizon, seed=30300 + c_spec.get("contract_idx", 1) * 41)
                net_events = pregenerate_network_events(c_spec.get("contract_idx", 1), rounds=horizon)

                for b_name in budgets:
                    b_conf = BUDGET_TIERS[b_name]
                    init_b = b_conf["budget_eth"]

                    print(f"\n---> MENGEKSEKUSI: {cname} | Rezim: {reg_name.upper()} | Anggaran: {b_name.upper()} ({init_b} ETH)")

                    K = len(c_spec["arms"])

                    # 1. UniformRandom (Forge Baseline)
                    p_random = UniformRandomPolicy(n_arms=K, context_dim=CONTEXT_DIM, initial_budget=init_b, total_horizon=horizon, seed=2026, name="UniformRandom")
                    # 2. LinUCB (Stationary Unconstrained)
                    p_linucb = DisjointLinUCB(n_arms=K, context_dim=CONTEXT_DIM, alpha=alpha, l2_lambda=L2_LAMBDA)
                    # 3. D-LinUCB (Discounted Unconstrained)
                    p_dlinucb = DiscountedLinUCB(n_arms=K, context_dim=CONTEXT_DIM, gamma=gamma, alpha=alpha, l2_lambda=L2_LAMBDA, decay_all_arms=True)
                    # 4. LinCBwK (Stationary Knapsack)
                    p_lincbwk = StationaryLinCBwK(n_arms=K, context_dim=CONTEXT_DIM, initial_budget=init_b, total_horizon=horizon, alpha=alpha, l2_lambda=L2_LAMBDA, seed=2026, name="LinCBwK")
                    # 5. D-LinCBwK (Discounted Knapsack - Proposed)
                    p_dlincbwk = DiscountedLinCBwK(n_arms=K, context_dim=CONTEXT_DIM, gamma=gamma, initial_budget=init_b, total_horizon=horizon, alpha=alpha, l2_lambda=L2_LAMBDA, seed=2026, name="D-LinCBwK")

                    policies = [
                        ("UniformRandom", p_random),
                        ("LinUCB", p_linucb),
                        ("D-LinUCB", p_dlinucb),
                        ("LinCBwK", p_lincbwk),
                        ("D-LinCBwK", p_dlincbwk),
                    ]

                    cell_results: Dict[str, Any] = {}
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
                        print(
                            f"     * {p_tag:<14}: Survived={res['rounds_survived']:2d} rds | "
                            f"Reward={res['cumulative_reward']:+5.2f} | Burned={res['capital_burned_eth']:.3f} ETH | "
                            f"CER={res['capital_efficiency_ratio']:5.1f} | Ruin={res['termination_reason']}"
                        )

                    experiment_record = {
                        "contract_id": c_spec["id"],
                        "contract_name": cname,
                        "tier": t_name,
                        "gas_regime": reg_name,
                        "budget_tier": b_name,
                        "initial_budget_eth": init_b,
                        "horizon_t": horizon,
                        "results": cell_results,
                    }
                    all_experiments.append(experiment_record)

    t_elapsed = time.time() - t_start
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = TELEMETRY_DIR / f"skripsi_experiments_{timestamp_str}.json"
    latest_file = TELEMETRY_DIR / "skripsi_experiments_latest.json"

    payload = {
        "timestamp": timestamp_str,
        "elapsed_seconds": round(t_elapsed, 2),
        "hyperparameters": {
            "alpha": alpha,
            "gamma": gamma,
            "l2_lambda": L2_LAMBDA,
            "horizon_t": horizon,
        },
        "total_experiments": len(all_experiments),
        "experiments": all_experiments,
    }

    out_file.write_text(json.dumps(payload, indent=2))
    latest_file.write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 96)
    print(f"🏁 EKSPERIMEN SKRIPSI SELESAI DALAM {t_elapsed:.2f} DETIK")
    print(f"   Total Evaluasi Grid: {len(all_experiments)} Matriks Kasus")
    print(f"   Telemetri Disimpan ke: {out_file}")
    print("=" * 96)

    return payload


def main():
    parser = argparse.ArgumentParser(description="Runner Eksperimen Skripsi Multi-Kondisi (Anvil EVM)")
    parser.add_argument("--tier", choices=["all", "shallow", "medium", "deep"], default="all", help="Kondisi 1: Kedalaman State")
    parser.add_argument("--regime", choices=["all", "stable", "volatile_spikes"], default="all", help="Kondisi 2: Dinamika Gas")
    parser.add_argument("--budget", choices=["all", "tight", "moderate", "loose"], default="all", help="Kondisi 3: Modal Gas B")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_T, help=f"Horizon T (default: {DEFAULT_HORIZON_T})")
    parser.add_argument("--contracts-limit", type=int, default=None, help="Batas jumlah kontrak per tier untuk quick run")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Hyperparameter Alpha (default: 0.15)")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Hyperparameter Gamma (default: 0.82)")

    args = parser.parse_args()
    run_experiment_matrix(
        tier=args.tier,
        regime=args.regime,
        budget=args.budget,
        horizon=args.horizon,
        contracts_limit=args.contracts_limit,
        alpha=args.alpha,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
