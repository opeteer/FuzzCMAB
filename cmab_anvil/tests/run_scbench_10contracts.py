#!/usr/bin/env python3
"""
SC-Bench Vault & Liquidity Organic Benchmark (11 Smart Contracts on Anvil)
Compares Disjoint LinUCB (gamma=1.0) vs Discounted LinUCB (gamma=0.82) under:
  1. Member-Level Standing (Strictly non-owner EOA, onboarded as a valid protocol member
     with initial token balances, ERC-20 approvals, and member registration to eliminate
     All-Revert Degeneracy without violating Zero-Admin assumptions).
  2. Pure Organic State Dynamics (No artificial regime shifts; dynamic on-chain state evolutions,
     EVM state transitions, balance changes, and storage slot dataflow).
  3. Stochastic Blockchain Network Events (Peer whale deposits, withdrawals, block time jitter, congestion)
     synchronized via Common Random Numbers (CRN / Paired Seeds) for 100% fairness.
  4. Dual Stopping Horizons (Fixed Horizon T=24 and Capital Exhaustion / Knapsack Stopping).
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

# Ensure repository root and cmab_anvil modules are importable
curr = Path(__file__).resolve()
REPO_ROOT = curr.parents[2] if len(curr.parents) > 2 else curr.parents[-1]
for p in curr.parents:
    if (p / "anvil_env").exists():
        REPO_ROOT = p
        break

CMAB_ROOT = REPO_ROOT / "cmab_anvil"
RESULTS_DIR = CMAB_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CMAB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CMAB_ROOT / "src"))

from cmab_anvil.algorithms.linucb import DisjointLinUCB
from cmab_anvil.algorithms.discounted_linucb import DiscountedLinUCB
from cmab_anvil.algorithms.uniform_random import UniformRandomPolicy
from cmab_anvil.algorithms.linucb_knapsack import StationaryLinCBwK
from cmab_anvil.algorithms.discounted_linucb_knapsack import DiscountedLinCBwK
from cmab_anvil.common.anvil_rpc import RegularUserAnvilClient
from cmab_anvil.Agen_Vault_Custody.context import (
    PublicContextExtractor,
    OnChainContextSnapshot,
    CONTEXT_DIM,
    FEATURE_NAMES,
)

SOLC_BIN = "/home/opeteer/.local/bin/solc"
CAST_BIN = "/home/opeteer/.foundry/bin/cast"
ANVIL_BIN = "/home/opeteer/.foundry/bin/anvil"
RPC_URL = "http://127.0.0.1:8545"

# Anvil Standard Accounts
DEPLOYER_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"  # Account #0 (Owner / Admin)
DEPLOYER_ADDR = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"

PEER_PK = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"      # Account #1 (Peer / Whale User)
PEER_ADDR = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"

AGENT_PK = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"     # Account #2 (Protocol Member EOA)
AGENT_ADDR = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"

DATASET_DIR = CMAB_ROOT / "datasets" / "SC-Bench" / "categorized_dataset" / "Vault_and_Liquidity"
if not DATASET_DIR.exists():
    DATASET_DIR = REPO_ROOT / "SC-Bench" / "categorized_dataset" / "Vault_and_Liquidity"
MOCK_TOKEN_SOL = REPO_ROOT / "anvil_env" / "src" / "MockToken.sol"
MAX_UINT256 = "115792089237316195423570985008687907853269984665640564039457584007913129639935"


def ensure_anvil_running() -> None:
    p = subprocess.run([CAST_BIN, "block-number", "--rpc-url", RPC_URL], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print("[*] Starting Anvil node on port 8545 (with --steps-tracing enabled)...")
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


COMPILED_ABIS: Dict[str, List[Dict[str, Any]]] = {}


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
    rpc_call("anvil_setNextBlockBaseFeePerGas", ["0x3b9aca00"])  # Reset to 1 Gwei for deploy
    ok, out, err = run_cast(["send", "--private-key", DEPLOYER_PK, "--gas-price", "5000000000", "--create", create_hex])
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


def impersonate_and_transfer_erc20(from_holder: str, token_addr: str, to_addr: str, amount_wei: str) -> None:
    """Used strictly during pre-test onboarding to distribute member tokens from a contract/treasury."""
    rpc_call("anvil_impersonateAccount", [from_holder])
    send_tx(DEPLOYER_PK, from_holder, "", [], 0.5)  # gas eth
    run_cast(["send", "--from", from_holder, "--unlocked", token_addr, "transfer(address,uint256)", to_addr, amount_wei])
    rpc_call("anvil_stopImpersonatingAccount", [from_holder])


# =============================================================================
# 11 SC-Bench Contract Specifications with Member Onboarding & 3 Regimes
# =============================================================================
CONTRACT_SPECS = [
    {
        "id": "SC-01",
        "split": "injected",
        "file": "ClashForStash-0x939b0425.sol",
        "contract": "ClashForStash",
        "category": "Vault_Custody_Yield",
        "description": "Token custody vault with min/max deposit caps, swap toggles, and member transfers",
        "arms": [
            {"id": 0, "name": "member_deposit_small", "sig": "deposit(uint256)", "args": ["100000000000000000"], "val": 0.0, "role": "inflow_primary"},
            {"id": 1, "name": "member_deposit_medium", "sig": "deposit(uint256)", "args": ["150000000000000000"], "val": 0.0, "role": "inflow_secondary"},
            {"id": 2, "name": "member_transfer_peer", "sig": "transfer(address,uint256)", "args": [PEER_ADDR, "50000000000000000"], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_increase_allowance", "sig": "increaseAllowance(address,uint256)", "args": [PEER_ADDR, "100000000000000000"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "unsigned_withdraw_probe", "sig": "withdraw(uint256,bytes)", "args": ["100000000000000000", "0x00"], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "vault_eth_fallback", "sig": "", "args": [], "val": 0.05, "role": "fallback_ops"},
        ],
    },
    {
        "id": "SC-02",
        "split": "injected",
        "file": "DrawTT-0x02D33D35.sol",
        "contract": "DrawTT",
        "constructor_sig": "constructor(string,string,string)",
        "constructor_args": ["DrawTTCustody", "DTT", "ipfs://vault/"],
        "category": "Escrow_PullPayment_Vault",
        "description": "Custodial NFT & PullPayment escrow contract with dynamic mint fee and async escrow settlement",
        "arms": [
            {"id": 0, "name": "member_mint_DynamicURI", "sig": "mint(string)", "args": ["DYNAMIC_URI"], "val": 0.02, "role": "inflow_primary"},
            {"id": 1, "name": "member_mint_lowfee", "sig": "mint(string)", "args": ["DYNAMIC_URI_LOW"], "val": 0.01, "role": "inflow_secondary"},
            {"id": 2, "name": "escrow_withdraw_payments", "sig": "withdrawPayments(address)", "args": [DEPLOYER_ADDR], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_set_approval_all", "sig": "setApprovalForAll(address,bool)", "args": [PEER_ADDR, "true"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "unauthorized_setCost_probe", "sig": "setCost(uint256)", "args": ["0"], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "duplicate_uri_probe", "sig": "mint(string)", "args": ["STATIC_DUPLICATE"], "val": 0.02, "role": "fallback_ops"},
        ],
    },
    {
        "id": "SC-03",
        "split": "injected",
        "file": "NoahSwapStake-0x0ff004b6.sol",
        "contract": "NoahSwapStake",
        "category": "Staking_Pool_Custody",
        "description": "Multi-asset staking custody with order ID slots, deputy renewal permissions, and ETH fallback",
        "arms": [
            {"id": 0, "name": "member_stake_eth_order", "sig": "stakeEth(uint256,address,uint256,uint256)", "args": ["DYNAMIC_ID", "0x0000000000000000000000000000000000000000", "50000000000000000", "2500000000"], "val": 0.05, "role": "inflow_primary"},
            {"id": 1, "name": "member_stake_erc20_order", "sig": "stake(uint256,address,uint256,uint256)", "args": ["DYNAMIC_ID_ERC20", "MOCK_TOKEN", "1000000000000000000", "2500000000"], "val": 0.0, "role": "inflow_secondary"},
            {"id": 2, "name": "deputy_order_renewal", "sig": "renewal(uint256[],uint256[])", "args": ["[1]", "[DYNAMIC_ENDTIME]"], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "deputy_set_signer", "sig": "setSigner(address)", "args": [AGENT_ADDR], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "admin_setDeputy_probe", "sig": "setDeputy(address)", "args": [AGENT_ADDR], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "pool_eth_fallback", "sig": "", "args": [], "val": 0.02, "role": "fallback_ops"},
        ],
    },
    {
        "id": "SC-04",
        "split": "injected",
        "file": "PyramidToken-0xbc69f076.sol",
        "contract": "PyramidToken",
        "category": "Liquidity_Pyramid_Vault",
        "description": "Tiered staking & liquidity vault with 30-day lockup maturity, referral bonus, and burn mechanics",
        "arms": [
            {"id": 0, "name": "member_stake_30days", "sig": "stake(uint256,uint256)", "args": ["1000000000000000000", "2592000"], "val": 0.0, "role": "inflow_primary"},
            {"id": 1, "name": "member_set_referrer", "sig": "setReferrer(address)", "args": [PEER_ADDR], "val": 0.0, "role": "inflow_secondary"},
            {"id": 2, "name": "member_withdraw_mature", "sig": "withdraw()", "args": [], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_burn_tokens", "sig": "burn(uint256)", "args": ["100000000000000000"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "member_transfer_peer", "sig": "transfer(address,uint256)", "args": [PEER_ADDR, "100000000000000000"], "val": 0.0, "role": "fallback_ops"},
            {"id": 5, "name": "invalid_short_stake_probe", "sig": "stake(uint256,uint256)", "args": ["100000000000000000", "60"], "val": 0.0, "role": "probe"},
        ],
    },
    {
        "id": "SC-05",
        "split": "injected",
        "file": "THREEAMIGOSCAPITAL-0x5bf05c34.sol",
        "contract": "THREEAMIGOSCAPITAL",
        "category": "Staking_Treasury",
        "description": "Capital staking treasury with lockup penalty before maturity, reward multiplier, and AMM limits",
        "arms": [
            {"id": 0, "name": "member_stake_short", "sig": "stake(uint256,uint256)", "args": ["1000000000000000000", "5"], "val": 0.0, "role": "inflow_primary"},
            {"id": 1, "name": "member_approve_router", "sig": "approve(address,uint256)", "args": [PEER_ADDR, "500000000000000000"], "val": 0.0, "role": "inflow_secondary"},
            {"id": 2, "name": "member_unstake_claim", "sig": "unstake()", "args": [], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_transfer_shares", "sig": "transfer(address,uint256)", "args": [PEER_ADDR, "100000000000000000"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "member_increase_allowance", "sig": "increaseAllowance(address,uint256)", "args": [DEPLOYER_ADDR, "100000000000000000"], "val": 0.0, "role": "fallback_ops"},
            {"id": 5, "name": "admin_setRules_probe", "sig": "setRules(address,uint256,uint8,uint8)", "args": [AGENT_ADDR, "0", "0", "0"], "val": 0.0, "role": "probe"},
        ],
    },
    {
        "id": "SC-06",
        "split": "origin",
        "file": "PymeStaking-0x90b8fbb7.sol",
        "contract": "PymeStaking",
        "category": "TimeLocked_Staking_Custody",
        "description": "Upgradeable staking custody with lockup tiers, premature unlock, and claim gates",
        "arms": [
            {"id": 0, "name": "member_stake_tier0", "sig": "stake(uint256,uint256)", "args": ["500000000000000000", "1"], "val": 0.0, "role": "inflow_primary"},
            {"id": 1, "name": "member_stake_tier1", "sig": "stake(uint256,uint256)", "args": ["1000000000000000000", "7"], "val": 0.0, "role": "inflow_secondary"},
            {"id": 2, "name": "member_claim_mature", "sig": "claim(uint256)", "args": ["0"], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_unlock_premature", "sig": "unlockPremature(uint256)", "args": ["1"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "admin_withdraw_probe", "sig": "withdraw()", "args": [], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "admin_setReward_probe", "sig": "setReward(uint256,uint256)", "args": ["1", "5000"], "val": 0.0, "role": "fallback_ops"},
        ],
    },
    {
        "id": "SC-07",
        "split": "injected",
        "file": "Token-0x7c81418a.sol",
        "contract": "Token",
        "category": "Liquidity_Vault_Token",
        "description": "Taxed liquidity token vault with public ETH flush, bot blacklist filter, and fee-bearing transfers",
        "arms": [
            {"id": 0, "name": "member_transfer_small", "sig": "transfer(address,uint256)", "args": [PEER_ADDR, "100000000000000000"], "val": 0.0, "role": "inflow_primary"},
            {"id": 1, "name": "member_transfer_medium", "sig": "transfer(address,uint256)", "args": [DEPLOYER_ADDR, "200000000000000000"], "val": 0.0, "role": "inflow_secondary"},
            {"id": 2, "name": "public_withdraw_tax_eth", "sig": "withdraw()", "args": [], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_approve_spender", "sig": "approve(address,uint256)", "args": [PEER_ADDR, "500000000000000000"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "admin_removeLimits_probe", "sig": "removeLimits()", "args": [], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "vault_eth_fallback", "sig": "", "args": [], "val": 0.03, "role": "fallback_ops"},
        ],
    },
    {
        "id": "SC-08",
        "split": "origin",
        "file": "ADWGenerator-0x544D5a9a.sol",
        "contract": "ADWGenerator",
        "category": "Fee_Custody_Vault",
        "description": "Tiered fee custody vault where creation fees dynamically adjust across protocol phases",
        "arms": [
            {"id": 0, "name": "member_submit_basic", "sig": "submit(uint256,uint256,bytes32)", "args": ["0", "1", "0x0000000000000000000000000000000000000000000000000000000000000001"], "val": 0.05, "role": "inflow_primary"},
            {"id": 1, "name": "member_submit_drippy", "sig": "submit(uint256,uint256,bytes32)", "args": ["1", "1", "0x0000000000000000000000000000000000000000000000000000000000000002"], "val": 0.08, "role": "inflow_secondary"},
            {"id": 2, "name": "member_submit_legendary", "sig": "submit(uint256,uint256,bytes32)", "args": ["2", "1", "0x0000000000000000000000000000000000000000000000000000000000000003"], "val": 0.15, "role": "secondary_ops"},
            {"id": 3, "name": "member_submit_mythic", "sig": "submit(uint256,uint256,bytes32)", "args": ["3", "1", "0x0000000000000000000000000000000000000000000000000000000000000004"], "val": 0.25, "role": "emergency_ops"},
            {"id": 4, "name": "admin_withdrawMoney_probe", "sig": "withdrawMoney()", "args": [], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "admin_toggleBurn_probe", "sig": "togglePublicBurnActive()", "args": [], "val": 0.0, "role": "fallback_ops"},
        ],
    },
    {
        "id": "SC-09",
        "split": "origin",
        "file": "AAMG-0x637aaa9c.sol",
        "contract": "AAMG",
        "category": "Pool_Custody_Mint",
        "description": "USDT-backed pool custody with dynamic ticket supply, mint pricing, and NFT approvals",
        "arms": [
            {"id": 0, "name": "member_public_mint_1", "sig": "publicMint(uint48)", "args": ["1"], "val": 0.0, "role": "inflow_primary"},
            {"id": 1, "name": "member_public_mint_2", "sig": "publicMint(uint48)", "args": ["2"], "val": 0.0, "role": "inflow_secondary"},
            {"id": 2, "name": "member_set_approval_all", "sig": "setApprovalForAll(address,bool)", "args": [PEER_ADDR, "true"], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_transfer_nft", "sig": "approve(address,uint256)", "args": [PEER_ADDR, "0"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "admin_withdraw_probe", "sig": "withdraw(uint256,address)", "args": ["100000000000000000", AGENT_ADDR], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "admin_devMint_probe", "sig": "devMint(address,uint48)", "args": [AGENT_ADDR, "1"], "val": 0.0, "role": "fallback_ops"},
        ],
    },
    {
        "id": "SC-10",
        "split": "origin",
        "file": "Shuffle-0x137110Eb.sol",
        "contract": "Shuffle",
        "category": "Liquidity_Pool_Lottery",
        "description": "Custodial bidding room pool with time-locked bid withdrawal and unprotected room modification",
        "arms": [
            {"id": 0, "name": "member_place_bid_room1", "sig": "placeBid(uint256)", "args": ["1"], "val": 0.05, "role": "inflow_primary"},
            {"id": 1, "name": "member_place_bid_room2", "sig": "placeBid(uint256)", "args": ["2"], "val": 0.08, "role": "inflow_secondary"},
            {"id": 2, "name": "member_withdraw_bid_room1", "sig": "withdrawBid(uint256)", "args": ["1"], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "unprotected_modify_room2", "sig": "modifyRoom(uint256,bool,uint256,address)", "args": ["2", "true", "5", AGENT_ADDR], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "admin_createRoom_probe", "sig": "createRoom(string,uint256,uint256,uint256,uint256,uint256,address)", "args": ["Hack", "100", "1000", "1", "5", "60", AGENT_ADDR], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "zero_bid_probe", "sig": "placeBid(uint256)", "args": ["1"], "val": 0.0, "role": "fallback_ops"},
        ],
    },
    {
        "id": "SC-11",
        "split": "origin",
        "file": "Airdrop-0x897b328c.sol",
        "contract": "Airdrop",
        "category": "Custodial_Merkle_Airdrop",
        "description": "Merkle-verified token distribution custody with single-claim lock and underlying token operations",
        "arms": [
            {"id": 0, "name": "member_merkle_claim_100", "sig": "claim(uint256,bytes32[])", "args": ["100000000000000000000", "[]"], "val": 0.0, "role": "inflow_primary"},
            {"id": 1, "name": "member_merkle_claim_250", "sig": "claim(uint256,bytes32[])", "args": ["250000000000000000000", "[]"], "val": 0.0, "role": "inflow_secondary"},
            {"id": 2, "name": "member_token_transfer", "target": "MOCK_TOKEN", "sig": "transfer(address,uint256)", "args": [PEER_ADDR, "10000000000000000000"], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_token_approve", "target": "MOCK_TOKEN", "sig": "approve(address,uint256)", "args": [PEER_ADDR, "50000000000000000000"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "admin_setRoot_probe", "sig": "setRoot(bytes32)", "args": ["0x0000000000000000000000000000000000000000000000000000000000000000"], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "admin_setToken_probe", "sig": "setToken(address)", "args": [AGENT_ADDR], "val": 0.0, "role": "fallback_ops"},
        ],
    },
]


def setup_member_environment(spec: Dict[str, Any], caddr: str, mock_token_addr: str) -> Dict[str, Any]:
    """
    Onboards AGENT_ADDR (Account #2) as a legitimate protocol member (initial token balances,
    allowances, and valid pool parameters) while strictly keeping DEPLOYER_ADDR (Account #0)
    as the sole contract owner.
    """
    cid = spec["id"]
    meta: Dict[str, Any] = {"mock_token": mock_token_addr}

    # Mint MockToken to Member (Agent), Peer, and Vault
    send_tx(DEPLOYER_PK, mock_token_addr, "mint(address,uint256)", [AGENT_ADDR, "100000000000000000000000"])
    send_tx(DEPLOYER_PK, mock_token_addr, "mint(address,uint256)", [PEER_ADDR, "100000000000000000000000"])
    send_tx(DEPLOYER_PK, mock_token_addr, "mint(address,uint256)", [caddr, "100000000000000000000000"])
    # Agent approves target contract to spend its MockToken
    send_tx(AGENT_PK, mock_token_addr, "approve(address,uint256)", [caddr, MAX_UINT256])

    if cid == "SC-01":  # ClashForStash
        send_tx(DEPLOYER_PK, caddr, "enableTrading()", [])
        send_tx(DEPLOYER_PK, caddr, "setDepositMinMax(uint256,uint256)", ["10000000000000000", "2000000000000000000"])
        send_tx(DEPLOYER_PK, caddr, "transfer(address,uint256)", [AGENT_ADDR, "50000000000000000000"])
        send_tx(DEPLOYER_PK, caddr, "", [], 3.0)

    elif cid == "SC-02":  # DrawTT
        send_tx(DEPLOYER_PK, caddr, "setCost(uint256)", ["10000000000000000"])  # 0.01 ETH
        send_tx(PEER_PK, caddr, "mint(string)", ["STATIC_DUPLICATE"], 0.02)

    elif cid == "SC-03":  # NoahSwapStake
        send_tx(DEPLOYER_PK, caddr, "", [], 3.0)
        send_tx(DEPLOYER_PK, caddr, "setDeputy(address)", [AGENT_ADDR])  # Member granted deputy operator role (not owner!)
        # Seed initial order #1 so renewal can work when unlocked
        send_tx(AGENT_PK, caddr, "stakeEth(uint256,address,uint256,uint256)", ["1", "0x0000000000000000000000000000000000000000", "50000000000000000", "2000000000"], 0.05)

    elif cid == "SC-04":  # PyramidToken
        # Distribute member PyramidTokens from initial treasury holder 0xfb8F827201f204c4Dd05aA692eDcE9771A0ab3e6
        impersonate_and_transfer_erc20(
            "0xfb8F827201f204c4Dd05aA692eDcE9771A0ab3e6",
            caddr,
            AGENT_ADDR,
            "5000000000000000000000",
        )
        send_tx(AGENT_PK, caddr, "approve(address,uint256)", [caddr, MAX_UINT256])

    elif cid == "SC-05":  # THREEAMIGOSCAPITAL
        send_tx(DEPLOYER_PK, caddr, "setRules(address,uint256,uint8,uint8)", [PEER_ADDR, "1000000000000000000000000", "1", "1"])
        send_tx(DEPLOYER_PK, caddr, "transfer(address,uint256)", [AGENT_ADDR, "5000000000000000000000"])

    elif cid == "SC-06":  # PymeStaking
        send_tx(DEPLOYER_PK, caddr, "initialize(address,address)", [mock_token_addr, DEPLOYER_ADDR])
        send_tx(DEPLOYER_PK, caddr, "setReward(uint256,uint256)", ["1", "500"])
        send_tx(DEPLOYER_PK, caddr, "setReward(uint256,uint256)", ["7", "1500"])

    elif cid == "SC-07":  # Token
        impersonate_and_transfer_erc20(caddr, caddr, AGENT_ADDR, "5000000000000000000000")
        send_tx(DEPLOYER_PK, caddr, "", [], 2.0)

    elif cid == "SC-08":  # ADWGenerator
        # Initially set creation fees low (0.04 ETH for basic/drippy, high for legendary/mythic)
        send_tx(
            DEPLOYER_PK,
            caddr,
            "setCreationFee(uint256,uint256,uint256,uint256)",
            ["40000000000000000", "70000000000000000", "500000000000000000", "500000000000000000"],
        )

    elif cid == "SC-09":  # AAMG
        send_tx(DEPLOYER_PK, caddr, "setUSDT(address)", [mock_token_addr])
        send_tx(DEPLOYER_PK, caddr, "setDayMintPrice(uint256)", ["1000000000000000000"])  # 1 USDT
        send_tx(DEPLOYER_PK, caddr, "setTiket(uint48)", ["50"])

    elif cid == "SC-10":  # Shuffle
        # Create Room #1 and Room #2 with valid minBid (0.02 ETH) and maxBid (2.0 ETH)
        send_tx(
            DEPLOYER_PK,
            caddr,
            "createRoom(string,uint256,uint256,uint256,uint256,uint256,address)",
            ["Room1", "20000000000000000", "2000000000000000000", "5", "10", "3600", DEPLOYER_ADDR],
        )
        send_tx(
            DEPLOYER_PK,
            caddr,
            "createRoom(string,uint256,uint256,uint256,uint256,uint256,address)",
            ["Room2", "20000000000000000", "2000000000000000000", "5", "10", "3600", DEPLOYER_ADDR],
        )

    elif cid == "SC-11":  # Airdrop
        send_tx(DEPLOYER_PK, caddr, "setToken(address)", [mock_token_addr])
        # Compute single-leaf Merkle root for AGENT_ADDR claiming 100e18 tokens
        _, packed, _ = run_cast(["abi-encode", "f(address,uint256)", AGENT_ADDR, "100000000000000000000"])
        # Note: solidity abi.encodePacked(address, uint256) is 20-byte address + 32-byte uint256
        addr_clean = AGENT_ADDR.lower().replace("0x", "")
        amt_hex = hex(100 * 10**18)[2:].zfill(64)
        _, leaf_hash, _ = run_cast(["keccak", "0x" + addr_clean + amt_hex])
        send_tx(DEPLOYER_PK, caddr, "setRoot(bytes32)", [leaf_hash])

    return meta


def pregenerate_network_events(contract_idx: int, rounds: int, event_mode: str = "stochastic") -> List[Dict[str, Any]]:
    """
    Pre-generates stochastic blockchain network events under Common Random Numbers (CRN):
      - 'none': Clean baseline with zero external disturbances.
      - 'stochastic': Full dynamic environment featuring:
          1. peer_liquidity (25%): External peer deposit/whale inflow.
          2. peer_withdrawal (25%): Peer withdraws their own legitimate deposited balance.
          3. block_jitter (20%): Time-lock/latency drift (30-900s).
          4. congestion_poke (15%): Background mempool token transfers.
          5. none (15%): Normal calm block interval.
    """
    if event_mode == "none":
        return [{"round": t, "type": "none", "eth_delta": 0.0, "time_delta": 0} for t in range(1, rounds + 1)]

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


def pregenerate_gas_prices(contract_idx: int, gas_mode: str, rounds: int) -> List[float]:
    """
    Pre-generates gas prices (in Gwei) under Common Random Numbers (CRN):
      - 'static': 1.0 Gwei constant (control baseline).
      - 'normal': Log-normal EIP-1559 dynamic between 8.0 and 32.0 Gwei (mean ~ 15 Gwei).
      - 'worst_case': Extreme congestion & gas war jump-diffusion between 60.0 and 500.0 Gwei.
    """
    if gas_mode == "static":
        return [1.0] * rounds

    rng = np.random.default_rng(seed=30300 + contract_idx * 41)
    if gas_mode == "normal":
        samples = rng.lognormal(mean=np.log(15.0), sigma=0.35, size=rounds)
        clipped = np.clip(np.round(samples, 2), 8.0, 32.0)
        return [float(x) for x in clipped]
    elif gas_mode == "worst_case":
        base = rng.uniform(60.0, 140.0, size=rounds)
        has_spike = rng.uniform(0, 1, size=rounds) < 0.25
        spikes = has_spike * (rng.pareto(a=1.8, size=rounds) * 80.0)
        clipped = np.clip(np.round(base + spikes, 2), 60.0, 500.0)
        return [float(x) for x in clipped]
    return [1.0] * rounds


def execute_network_event(
    ev: Dict[str, Any],
    caddr: str,
    mock_token_addr: str,
    peer_state: Dict[str, float],
    client: RegularUserAnvilClient,
) -> None:
    ev_type = ev["type"]
    if ev_type == "peer_liquidity":
        # Peer sends ETH to the vault (inflow shock)
        send_tx(PEER_PK, caddr, "", [], ev["eth_delta"])
        peer_state["deposited_eth"] = peer_state.get("deposited_eth", 0.0) + ev["eth_delta"]

    elif ev_type == "peer_withdrawal":
        # Peer withdraws legitimate funds strictly limited to what they own in the vault!
        avail = peer_state.get("deposited_eth", 0.0)
        c_bal = client.get_eth_balance(caddr)
        max_withdrawable = min(avail, c_bal)
        if max_withdrawable > 0.01:
            w_amt = float(np.round(min(max_withdrawable, ev["eth_delta"]), 4))
            peer_state["deposited_eth"] -= w_amt
            # Execute on-chain withdrawal from vault back to PEER_ADDR
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
    client: RegularUserAnvilClient,
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

    # Unified 4-Tier resolution (Dynamic ABI -> Multi-Standard ABI -> Keccak256 Mapping Slot -> Companion Token)
    agent_bal = client.query_agent_stake(
        caddr,
        AGENT_ADDR,
        fallback_token_address=mock_token_addr,
    )

    # Unified 4-Tier resolution (Dynamic ABI -> Multi-Standard ABI -> Raw Storage Slot Inference)
    total_dep = client.query_total_deposited(
        caddr,
        vault_eth_hint=vault_eth,
    )

    token_bal = client.call_uint256_ether_with_fallbacks(
        mock_token_addr, ["balanceOf(address)(uint256)"], AGENT_ADDR
    )
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

    # Resolve dynamic arguments (e.g., unique order IDs or URIs for member calls)
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
    anomalies: List[str] = []
    role = arm.get("role", "")

    # Canonical MAB non-negative boundary: Reverted transactions yield 0.0 (null utility)
    if not tx_succeeded:
        return 0.0, anomalies

    # Base reward for valid on-chain execution
    reward = 0.20

    # Organic role-based utility without artificial regime payoff inversions
    if role in ("inflow_primary", "inflow_secondary"):
        reward += 0.40
    elif role == "secondary_ops":
        reward += 0.35
    elif role == "emergency_ops":
        reward += 0.20
    elif role == "fallback_ops":
        reward += 0.15

    # Bonus for regular user access control bypass (Security Finding)
    if role == "probe" or "unprotected" in arm["name"] or "public_withdraw" in arm["name"]:
        reward += 0.20
        anomalies.append(f"SECURITY FINDING: Member EOA executed sensitive function '{arm['name']}' ({arm['sig']})")

    # Bonus for positive net ETH extraction
    eth_gain = next_snap.agent_eoa_eth_balance - prev_snap.agent_eoa_eth_balance
    if eth_gain > 0.005:
        reward += min(0.20, eth_gain * 0.5)
        anomalies.append(f"Extracted +{eth_gain:.3f} ETH on-chain via {arm['name']}")

    # Strictly clip to [0.0, 1.0]
    final_reward = float(np.clip(reward, 0.0, 1.0))
    return final_reward, anomalies


def run_policy_on_contract(
    policy_name: str,
    policy: Any,
    spec: Dict[str, Any],
    contract_bytecode: str,
    mock_token_bytecode: str,
    network_events: List[Dict[str, Any]],
    gas_prices: Optional[List[float]] = None,
    gas_mode: str = "static",
    scenario: str = "fixed",
    fixed_rounds: int = 24,
    initial_capital_eth: float = 0.50,
    min_capital_threshold_eth: float = 0.005,
    max_safety_rounds: int = 10000,
) -> Dict[str, Any]:
    # 1. Fresh deployment of MockToken and Target Contract
    mock_token_addr = deploy_bytecode(mock_token_bytecode, "constructor(uint256)", ["1000000000000000000000000"])
    caddr = deploy_bytecode(
        contract_bytecode,
        spec.get("constructor_sig"),
        spec.get("constructor_args"),
    )
    if not caddr or not mock_token_addr:
        raise RuntimeError(f"Failed to deploy {spec['contract']}")

    # 2. Setup Member Onboarding (Initial Token Balances, Approvals, Pool Init)
    setup_member_environment(spec, caddr, mock_token_addr)

    # Seed peer's initial deposit into vault so peer owns legitimate funds ready to be withdrawn
    send_tx(PEER_PK, caddr, "", [], 0.50)
    peer_state: Dict[str, float] = {"deposited_eth": 0.50}

    # 3. Setup Initial Agent Capital for Scenario
    client = RegularUserAnvilClient(rpc_url=RPC_URL)
    if spec["contract"] in COMPILED_ABIS:
        client.register_contract_abi(caddr, COMPILED_ABIS[spec["contract"]])
    starting_capital = initial_capital_eth if scenario == "capital" else 5.0
    rpc_call("anvil_setBalance", [AGENT_ADDR, hex(int(starting_capital * 1e18))])

    arms = spec["arms"]
    K = len(arms)
    pos_vals = [float(a.get("val", 0.0)) for a in arms if float(a.get("val", 0.0)) > 0.0]
    min_dep_req = min(pos_vals) if pos_vals else 0.05
    extractor = PublicContextExtractor(
        min_deposit_eth=min_dep_req,
        vault_cap_eth=20.0,
        agent_init_eth=starting_capital,
    )

    rewards: List[float] = []
    round_logs: List[Dict[str, Any]] = []
    arm_counts = [0] * K
    success_count = 0
    anomalies_found: List[str] = []
    recent_reverts: List[int] = []
    last_sstore_flag = False
    last_waypoint_flag = False

    target_rounds = max_safety_rounds if scenario == "capital" else fixed_rounds
    termination_reason = "completed"

    for t in range(1, target_rounds + 1):
        # In Capital Scenario, check if agent is bankrupt before round
        current_bal = client.get_eth_balance(AGENT_ADDR)
        if scenario == "capital" and current_bal <= min_capital_threshold_eth:
            termination_reason = "bankruptcy"
            break

        # Dynamic On-Chain Gas Price Setting (Exogenous friction without altering agent's context vector!)
        current_gas_gwei = gas_prices[t - 1] if gas_prices and (t - 1 < len(gas_prices)) else 1.0
        rpc_call("anvil_setNextBlockBaseFeePerGas", [hex(int(current_gas_gwei * 1e9))])

        # Replay pre-generated stochastic blockchain event (identical for both algorithms)
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

        tx_ok, tx_out = execute_member_arm(
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

        # Query post-tx balance and transaction gas cost
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
            "burned_eth": round(max(0.0, starting_capital - post_bal), 6),
            "event_type": ev_name,
        }
        round_logs.append(round_log)

        # Knapsack policy internal stopping condition (if budget depleted)
        if hasattr(policy, "check_stopping_condition"):
            stopped, stop_reason = policy.check_stopping_condition(vec)
            if stopped:
                termination_reason = stop_reason
                break

        # In Capital Scenario, check if post-tx balance depleted
        if scenario == "capital" and post_bal <= min_capital_threshold_eth:
            termination_reason = "bankruptcy"
            break

    if scenario == "capital" and termination_reason == "completed" and len(rewards) >= max_safety_rounds:
        termination_reason = "max_horizon"

    actual_rounds = len(rewards) if rewards else 1
    final_capital = client.get_eth_balance(AGENT_ADDR)
    capital_burned = max(0.0, starting_capital - final_capital)
    cum_reward = float(np.sum(rewards)) if rewards else 0.0
    cap_efficiency = round(cum_reward / max(0.001, capital_burned), 3)

    # Compute Shannon exploration entropy of arm distribution
    probs = np.array(arm_counts, dtype=float) / float(actual_rounds)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0

    return {
        "address": caddr,
        "scenario": scenario,
        "gas_mode": gas_mode,
        "rounds_survived": actual_rounds,
        "termination_reason": termination_reason,
        "bankruptcy": (termination_reason == "bankruptcy"),
        "starting_capital_eth": starting_capital,
        "final_capital_eth": round(final_capital, 4),
        "capital_burned_eth": round(capital_burned, 4),
        "capital_efficiency_ratio": cap_efficiency,
        "cumulative_reward": round(cum_reward, 3),
        "mean_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
        "success_rate": round(float(success_count) / float(actual_rounds), 4),
        "revert_count": actual_rounds - success_count,
        "arm_counts": arm_counts,
        "arm_entropy_bits": round(entropy, 4),
        "theta_norms": [round(float(np.linalg.norm(policy.theta[a])), 4) for a in range(K)],
        "anomalies": sorted(list(set(anomalies_found))),
        "round_logs": round_logs,
    }


def run_benchmark_for_scenario(
    scenario: str,
    gas_mode: str = "static",
    event_mode: str = "stochastic",
    capital: float = 0.50,
    fixed_rounds: int = 24,
    max_safety_rounds: int = 10000,
    contracts_limit: Optional[int] = None,
    alpha: float = 0.15,
    use_theoretical_beta: bool = False,
) -> List[Dict[str, Any]]:
    print("=" * 96)
    specs_to_run = CONTRACT_SPECS[:contracts_limit] if contracts_limit else CONTRACT_SPECS
    total_c = len(specs_to_run)
    beta_tag = "THEORETICAL_BETA" if use_theoretical_beta else "MATCHED_ALPHA"
    if scenario == "fixed":
        print(f"🚀 SCENARIO A: FIXED TIME HORIZON | Gas: {gas_mode.upper()} | Events: {event_mode.upper()} | Alpha: {alpha} ({beta_tag}) ({total_c} KONTRAK ON ANVIL)")
        print(f"   Setup: Member Privilege + Paired CRN | Fixed T = {fixed_rounds} Rounds/Contract")
    else:
        print(f"🚀 SCENARIO B: CAPITAL EXHAUSTION | Gas: {gas_mode.upper()} | Events: {event_mode.upper()} | Alpha: {alpha} ({beta_tag}) (KNAPSACK STOPPING ON ANVIL)")
        print(f"   Setup: Initial Capital = {capital:.2f} ETH | Stop: Balance <= 0.005 ETH OR T_max = {max_safety_rounds} ({total_c} Kontrak)")
    print("=" * 96)

    mock_token_bin = compile_contract(MOCK_TOKEN_SOL, "MockToken")
    if not mock_token_bin:
        raise RuntimeError("Failed to compile MockToken.sol")

    all_results = []
    t0 = time.time()
    event_horizon = max_safety_rounds if scenario == "capital" else fixed_rounds

    for idx, spec in enumerate(specs_to_run, 1):
        cname = spec["contract"]
        fpath = DATASET_DIR / spec["split"] / spec["file"]
        print(f"\n[{idx:02d}/{total_c:02d}] Kontrak: {cname:<20} ({spec['split']}/{spec['file']})")

        c_bin = compile_contract(fpath, cname)
        if not c_bin:
            print(f"       [!] Gagal kompilasi {cname}")
            continue

        net_events = pregenerate_network_events(idx, rounds=event_horizon, event_mode=event_mode)
        net_gas_prices = pregenerate_gas_prices(idx, gas_mode=gas_mode, rounds=event_horizon)

        K = len(spec["arms"])
        rand_policy = UniformRandomPolicy(
            n_arms=K, context_dim=CONTEXT_DIM, initial_budget=capital, total_horizon=event_horizon, seed=2026 + idx * 7, name="UniformRandom"
        )
        lin_policy = DisjointLinUCB(n_arms=K, context_dim=CONTEXT_DIM, alpha=alpha, l2_lambda=1.0)
        dlin_policy = DiscountedLinUCB(
            n_arms=K,
            context_dim=CONTEXT_DIM,
            gamma=0.82,
            alpha=alpha,
            l2_lambda=1.0,
            use_theoretical_beta=use_theoretical_beta,
            decay_all_arms=True,
        )
        lincbwk_policy = StationaryLinCBwK(
            n_arms=K,
            context_dim=CONTEXT_DIM,
            initial_budget=capital,
            total_horizon=event_horizon,
            alpha=alpha,
            l2_lambda=1.0,
            seed=2026 + idx * 7,
            name="LinCBwK",
        )
        dlincbwk_policy = DiscountedLinCBwK(
            n_arms=K,
            context_dim=CONTEXT_DIM,
            gamma=0.82,
            initial_budget=capital,
            total_horizon=event_horizon,
            alpha=alpha,
            l2_lambda=1.0,
            seed=2026 + idx * 7,
            name="D-LinCBwK",
        )

        rand_res = run_policy_on_contract(
            "UniformRandom", rand_policy, spec, c_bin, mock_token_bin, net_events,
            gas_prices=net_gas_prices, gas_mode=gas_mode,
            scenario=scenario, fixed_rounds=fixed_rounds, initial_capital_eth=capital, max_safety_rounds=max_safety_rounds,
        )
        lin_res = run_policy_on_contract(
            "LinUCB", lin_policy, spec, c_bin, mock_token_bin, net_events,
            gas_prices=net_gas_prices, gas_mode=gas_mode,
            scenario=scenario, fixed_rounds=fixed_rounds, initial_capital_eth=capital, max_safety_rounds=max_safety_rounds,
        )
        dlin_res = run_policy_on_contract(
            "D-LinUCB", dlin_policy, spec, c_bin, mock_token_bin, net_events,
            gas_prices=net_gas_prices, gas_mode=gas_mode,
            scenario=scenario, fixed_rounds=fixed_rounds, initial_capital_eth=capital, max_safety_rounds=max_safety_rounds,
        )
        lincbwk_res = run_policy_on_contract(
            "LinCBwK", lincbwk_policy, spec, c_bin, mock_token_bin, net_events,
            gas_prices=net_gas_prices, gas_mode=gas_mode,
            scenario=scenario, fixed_rounds=fixed_rounds, initial_capital_eth=capital, max_safety_rounds=max_safety_rounds,
        )
        dlincbwk_res = run_policy_on_contract(
            "D-LinCBwK", dlincbwk_policy, spec, c_bin, mock_token_bin, net_events,
            gas_prices=net_gas_prices, gas_mode=gas_mode,
            scenario=scenario, fixed_rounds=fixed_rounds, initial_capital_eth=capital, max_safety_rounds=max_safety_rounds,
        )

        entry = {
            "id": spec["id"],
            "contract": cname,
            "file": spec["file"],
            "category": spec["category"],
            "description": spec["description"],
            "scenario": scenario,
            "gas_mode": gas_mode,
            "event_mode": event_mode,
            "uniform_random": rand_res,
            "linucb": lin_res,
            "d_linucb": dlin_res,
            "lincbwk": lincbwk_res,
            "d_lincbwk": dlincbwk_res,
        }
        all_results.append(entry)

        if scenario == "fixed":
            print(
                f"       - Uniform  (Forge Base): Reward={rand_res['cumulative_reward']:+5.2f} | "
                f"Mean={rand_res['mean_reward']:.4f} | Succ={rand_res['success_rate']*100:4.1f}% | "
                f"Entropy={rand_res['arm_entropy_bits']:.2f}b | Arms={rand_res['arm_counts']}"
            )
            print(
                f"       - LinUCB   (gamma=1.00): Reward={lin_res['cumulative_reward']:+5.2f} | "
                f"Mean={lin_res['mean_reward']:.4f} | Succ={lin_res['success_rate']*100:4.1f}% | "
                f"Entropy={lin_res['arm_entropy_bits']:.2f}b | Arms={lin_res['arm_counts']}"
            )
            print(
                f"       - D-LinUCB (gamma=0.82): Reward={dlin_res['cumulative_reward']:+5.2f} | "
                f"Mean={dlin_res['mean_reward']:.4f} | Succ={dlin_res['success_rate']*100:4.1f}% | "
                f"Entropy={dlin_res['arm_entropy_bits']:.2f}b | Arms={dlin_res['arm_counts']}"
            )
            print(
                f"       - LinCBwK  (gamma=1.00): Reward={lincbwk_res['cumulative_reward']:+5.2f} | "
                f"Mean={lincbwk_res['mean_reward']:.4f} | Succ={lincbwk_res['success_rate']*100:4.1f}% | "
                f"Entropy={lincbwk_res['arm_entropy_bits']:.2f}b | Arms={lincbwk_res['arm_counts']}"
            )
            print(
                f"       - D-LinCBwK(gamma=0.82): Reward={dlincbwk_res['cumulative_reward']:+5.2f} | "
                f"Mean={dlincbwk_res['mean_reward']:.4f} | Succ={dlincbwk_res['success_rate']*100:4.1f}% | "
                f"Entropy={dlincbwk_res['arm_entropy_bits']:.2f}b | Arms={dlincbwk_res['arm_counts']}"
            )
        else:
            print(
                f"       - Uniform  (Forge Base): Survived={rand_res['rounds_survived']:2d} rds | "
                f"Burned={rand_res['capital_burned_eth']:.3f} ETH | CER={rand_res['capital_efficiency_ratio']:5.1f} | "
                f"Ruin={rand_res['termination_reason']} | Reward={rand_res['cumulative_reward']:+5.2f}"
            )
            print(
                f"       - LinUCB   (gamma=1.00): Survived={lin_res['rounds_survived']:2d} rds | "
                f"Burned={lin_res['capital_burned_eth']:.3f} ETH | CER={lin_res['capital_efficiency_ratio']:5.1f} | "
                f"Ruin={lin_res['termination_reason']} | Reward={lin_res['cumulative_reward']:+5.2f}"
            )
            print(
                f"       - D-LinUCB (gamma=0.82): Survived={dlin_res['rounds_survived']:2d} rds | "
                f"Burned={dlin_res['capital_burned_eth']:.3f} ETH | CER={dlin_res['capital_efficiency_ratio']:5.1f} | "
                f"Ruin={dlin_res['termination_reason']} | Reward={dlin_res['cumulative_reward']:+5.2f}"
            )
            print(
                f"       - LinCBwK  (gamma=1.00): Survived={lincbwk_res['rounds_survived']:2d} rds | "
                f"Burned={lincbwk_res['capital_burned_eth']:.3f} ETH | CER={lincbwk_res['capital_efficiency_ratio']:5.1f} | "
                f"Ruin={lincbwk_res['termination_reason']} | Reward={lincbwk_res['cumulative_reward']:+5.2f}"
            )
            print(
                f"       - D-LinCBwK(gamma=0.82): Survived={dlincbwk_res['rounds_survived']:2d} rds | "
                f"Burned={dlincbwk_res['capital_burned_eth']:.3f} ETH | CER={dlincbwk_res['capital_efficiency_ratio']:5.1f} | "
                f"Ruin={dlincbwk_res['termination_reason']} | Reward={dlincbwk_res['cumulative_reward']:+5.2f}"
            )

    elapsed = time.time() - t0
    fname = f"scbench_{scenario}_{gas_mode}_{event_mode}_results.json"
    out_path = RESULTS_DIR / fname
    out_path.write_text(json.dumps(all_results, indent=2))
    print("\n" + "=" * 96)
    print(f"🏁 SELESAI ({scenario.upper()} - {gas_mode.upper()} - {event_mode.upper()}) DALAM {elapsed:.2f} DETIK | Telemetri disimpan ke: {out_path}")
    print("=" * 96)
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SC-Bench CMAB Simulation on Anvil (Dual Stopping Scenarios & Gas Price Fluctuation)")
    parser.add_argument("--scenario", choices=["fixed", "capital", "both"], default="fixed", help="Stopping condition scenario")
    parser.add_argument("--gas-mode", choices=["static", "normal", "worst_case", "all"], default="static", help="Gas price fluctuation mode")
    parser.add_argument("--event-mode", choices=["none", "stochastic", "both"], default="stochastic", help="Blockchain network event mode (Ablation)")
    parser.add_argument("--capital", type=float, default=0.50, help="Initial capital in ETH for capital exhaustion scenario")
    parser.add_argument("--fixed-rounds", type=int, default=24, help="Fixed horizon rounds for scenario A")
    parser.add_argument("--max-rounds", type=int, default=10000, help="Max safety rounds for scenario B (default: 10000)")
    parser.add_argument("--contracts-limit", type=int, default=None, help="Limit number of contracts to evaluate (e.g. 5)")
    parser.add_argument("--alpha", type=float, default=0.15, help="Exploration hyperparameter alpha (default: 0.15)")
    parser.add_argument("--use-theoretical-beta", action="store_true", help="Enable theoretical confidence radius beta for D-LinUCB (default: False, matches alpha)")
    args = parser.parse_args()

    ensure_anvil_running()

    scenarios_to_run = ["fixed", "capital"] if args.scenario == "both" else [args.scenario]
    gas_modes_to_run = ["static", "normal", "worst_case"] if args.gas_mode == "all" else [args.gas_mode]
    event_modes_to_run = ["none", "stochastic"] if args.event_mode == "both" else [args.event_mode]

    for sc in scenarios_to_run:
        for gm in gas_modes_to_run:
            for em in event_modes_to_run:
                if sc == "fixed":
                    run_benchmark_for_scenario(
                        "fixed",
                        gas_mode=gm,
                        event_mode=em,
                        fixed_rounds=args.fixed_rounds,
                        contracts_limit=args.contracts_limit,
                        alpha=args.alpha,
                        use_theoretical_beta=args.use_theoretical_beta,
                    )
                else:
                    run_benchmark_for_scenario(
                        "capital",
                        gas_mode=gm,
                        event_mode=em,
                        capital=args.capital,
                        max_safety_rounds=args.max_rounds,
                        contracts_limit=args.contracts_limit,
                        alpha=args.alpha,
                        use_theoretical_beta=args.use_theoretical_beta,
                    )


if __name__ == "__main__":
    main()
