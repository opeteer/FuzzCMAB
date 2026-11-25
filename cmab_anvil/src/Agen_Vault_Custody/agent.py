"""
Agen_Vault_Custody: Environment & Agent Orchestrator for Custody Vault & ERC-20 Contracts.

Connects the domain-specific Vault/Token modeling (`context.py`, `arms.py`, `reward.py`)
with the universal CMAB strategy templates (`cmab_anvil/algorithms/linucb.py` and
`cmab_anvil/algorithms/discounted_linucb.py`) and universal Anvil RPC client
(`cmab_anvil/common/anvil_rpc.py`).
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import numpy as np

from ..algorithms.base import BaseCMABPolicy, ArmSelectionResult
from ..algorithms.linucb import DisjointLinUCB
from ..algorithms.discounted_linucb import DiscountedLinUCB
from ..algorithms.linucb_knapsack import StationaryLinCBwK
from ..algorithms.discounted_linucb_knapsack import DiscountedLinCBwK
from ..algorithms.uniform_random import UniformRandomPolicy
from ..common.anvil_rpc import UnprivilegedAnvilClient
from ..common.base_env import Arm, BaseUnprivilegedBanditEnv, StepTransition

from .context import (
    OnChainContextSnapshot,
    PublicContextExtractor,
    CONTEXT_DIM,
    FEATURE_NAMES,
)
from .arms import build_vault_custody_arm_catalog
from .reward import BanditRewardBreakdown, VaultCustodyRewardOracle


_GAS_USED_RE = re.compile(r"gasUsed\s+(\d+|0x[0-9a-fA-F]+)")


def generate_vault_gas_series(regime: str, rounds: int, seed: int = 42) -> List[float]:
    """
    Membangkitkan deret harga gas (Gwei) untuk putaran t = 1..T sesuai Tabel 3.1:
      - 'stable': Log-normal EIP-1559 (mu=ln(15), sigma=0.35), clipped [8.0, 32.0] Gwei
      - 'volatile_spikes': Uniform [60, 140] Gwei + 25% lonjakan Pareto (alpha=1.8 * 80), clipped [60.0, 500.0] Gwei
      - 'static': Konstan 1.0 Gwei
    """
    n = max(1, int(rounds))
    reg = regime.lower().strip()
    if reg == "static":
        return [1.0] * n
    rng = np.random.default_rng(seed)
    if reg in ("stable", "low_volatility", "normal"):
        samples = rng.lognormal(mean=np.log(15.0), sigma=0.35, size=n)
        clipped = np.clip(np.round(samples, 2), 8.0, 32.0)
        return [float(x) for x in clipped]
    if reg in ("volatile_spikes", "high_volatility", "worst_case"):
        base = rng.uniform(60.0, 140.0, size=n)
        has_spike = rng.uniform(0.0, 1.0, size=n) < 0.25
        spikes = has_spike * (rng.pareto(a=1.8, size=n) * 80.0)
        clipped = np.clip(np.round(base + spikes, 2), 60.0, 500.0)
        return [float(x) for x in clipped]
    raise ValueError(f"Rezim gas tidak dikenal: '{regime}'. Pilih 'stable', 'volatile_spikes', atau 'static'.")


def generate_external_party_events(rounds: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Membangkitkan kejadian transaksi pihak luar (PEER_USER_PK) berbasis Common Random Numbers (CRN):
      - 'peer_deposit': Pihak luar menyetor ETH ke VulnerableVault
      - 'peer_withdraw': Pihak luar menarik sebagian simpanan sahnya dari VulnerableVault
      - 'peer_token_transfer': Pihak luar mentransfer MockToken di jaringan
      - 'none': Tidak ada aktivitas pihak luar pada putaran tersebut
    """
    n = max(1, int(rounds))
    rng = np.random.default_rng(seed + 1009)
    events: List[Dict[str, Any]] = []
    for t in range(1, n + 1):
        ev_type = str(
            rng.choice(
                ["peer_deposit", "peer_withdraw", "peer_token_transfer", "none"],
                p=[0.30, 0.25, 0.20, 0.25],
            )
        )
        eth_amt = float(np.round(rng.uniform(0.05, 0.20), 3))
        events.append({"round": t, "type": ev_type, "eth_amount": eth_amt})
    return events


class VaultCustodyEnv(BaseUnprivilegedBanditEnv):
    """
    Concrete CMAB environment for `Agen_Vault_Custody` interacting with
    `VulnerableVault` and `MockToken` on Anvil without any special privileges,
    equipped with dynamic gas price regimes and external party transaction simulation.
    """

    def __init__(
        self,
        rpc_url: str = "http://127.0.0.1:8545",
        deployed_json_path: Optional[Path] = None,
        agent_ref_budget_eth: float = 0.50,
        gas_regime: str = "stable",
        horizon_t: int = 50,
        enable_peer_traffic: bool = True,
        min_capital_threshold_eth: float = 0.005,
        seed: int = 42,
    ):
        rpc_client = UnprivilegedAnvilClient(rpc_url=rpc_url)
        arms = build_vault_custody_arm_catalog()
        super().__init__(rpc_client=rpc_client, arms=arms, revert_window=8)

        curr = Path(__file__).resolve()
        repo_root = curr.parents[3] if len(curr.parents) > 3 else curr.parents[-1]
        for p in curr.parents:
            if (p / "anvil_env").exists():
                repo_root = p
                break
        self.deployed_json_path = (
            deployed_json_path
            if deployed_json_path is not None
            else repo_root / "anvil_env" / "deployed_addresses.json"
        )
        self.initial_budget_eth: float = float(agent_ref_budget_eth)
        self.remaining_budget_eth: float = float(agent_ref_budget_eth)
        self.min_capital_threshold_eth: float = float(min_capital_threshold_eth)
        self.gas_regime: str = gas_regime
        self.horizon_t: int = max(1, int(horizon_t))
        self.enable_peer_traffic: bool = bool(enable_peer_traffic)
        self.seed: int = int(seed)

        self.gas_series: List[float] = generate_vault_gas_series(
            self.gas_regime, rounds=max(self.horizon_t, 200), seed=self.seed
        )
        self.external_events: List[Dict[str, Any]] = generate_external_party_events(
            rounds=max(self.horizon_t, 200), seed=self.seed
        )
        self.current_gas_gwei: float = self.gas_series[0]
        self.last_external_event: str = "none"

        self.context_extractor = PublicContextExtractor(
            min_deposit_eth=0.25,
            vault_cap_eth=15.0,
            agent_init_eth=agent_ref_budget_eth,
        )
        self.reward_oracle = VaultCustodyRewardOracle()

        self.vault_address: str = ""
        self.token_address: str = ""
        self.last_sstore_flag: bool = False
        self.last_waypoint_flag: bool = False
        self._load_contract_addresses()

    def _load_contract_addresses(self) -> None:
        if not self.deployed_json_path.exists():
            raise FileNotFoundError(
                f"Deployed addresses file not found at {self.deployed_json_path}. "
                "Run ./anvil_env/scripts/run_fair_simulation.sh first."
            )
        data = json.loads(self.deployed_json_path.read_text())
        self.vault_address = data["VulnerableVault"]
        self.token_address = data["MockToken"]

        # Tier 2: Auto-register Foundry compiled ABI if available for dynamic introspection
        vault_abi_path = (
            self.deployed_json_path.parent
            / "out"
            / "VulnerableVault.sol"
            / "VulnerableVault.json"
        )
        if vault_abi_path.exists():
            try:
                artifact = json.loads(vault_abi_path.read_text())
                if isinstance(artifact.get("abi"), list):
                    self.rpc_client.register_contract_abi(
                        self.vault_address, artifact["abi"]
                    )
            except Exception:
                pass

    def verify_unprivileged_standing(self) -> Dict[str, str]:
        info = self.rpc_client.verify_not_owner(
            contract_address=self.vault_address,
            owner_sig="owner()(address)",
        )
        return {
            "agent_name": "Agen_Vault_Custody",
            "agent_address": info["agent_address"],
            "vault_owner": info["contract_owner"],
            "vault_address": self.vault_address,
            "token_address": self.token_address,
            "standing": info["standing"],
        }

    def observe_snapshot(self) -> OnChainContextSnapshot:
        c = self.rpc_client
        block_number = c.get_block_number()
        vault_eth = c.get_eth_balance(self.vault_address)
        agent_eoa_eth = c.get_eth_balance(c.agent_address)

        # Unified 4-Tier resolution (Dynamic ABI -> Multi-Standard ABI -> Raw Storage Slot Inference)
        vault_total_dep = c.query_total_deposited(
            self.vault_address, vault_eth_hint=vault_eth
        )

        # Unified 4-Tier resolution (Dynamic ABI -> Multi-Standard ABI -> Keccak256 Mapping Slot Inference)
        agent_vault_bal = c.query_agent_stake(
            self.vault_address, c.agent_address
        )

        solv_ok, solv_val = c.call_bool(self.vault_address, "check_solvency()(bool)")
        if not solv_ok:
            # Generic solvency check fallback when `check_solvency()` is absent
            solv_val = (vault_eth + 1e-9 >= vault_total_dep)

        _, locked_val = c.call_bool(self.vault_address, "isLocked()(bool)")
        code_alive = c.get_code_alive(self.vault_address)
        agent_token_bal = c.call_uint256_ether_with_fallbacks(
            self.token_address,
            ["balanceOf(address)(uint256)"],
            c.agent_address,
        )

        return OnChainContextSnapshot(
            block_number=block_number,
            vault_eth_balance=vault_eth,
            vault_total_deposited=vault_total_dep,
            agent_vault_balance=agent_vault_bal,
            agent_eoa_eth_balance=agent_eoa_eth,
            agent_token_balance=agent_token_bal,
            vault_is_solvent=solv_val,
            vault_is_locked=locked_val,
            vault_code_alive=code_alive,
            recent_revert_rate=self.recent_revert_rate,
            sstore_executed=self.last_sstore_flag,
            dataflow_waypoint=self.last_waypoint_flag,
            min_deposit_requirement=self.context_extractor.min_deposit_eth,
        )

    def observe_context(self) -> Tuple[np.ndarray, OnChainContextSnapshot]:
        snap = self.observe_snapshot()
        vec = self.context_extractor.to_vector(snap)
        return vec, snap

    def _set_network_base_fee(self, gas_price_gwei: float) -> None:
        """
        Mengatur base fee jaringan pada node simulator Anvil sebelum putaran t
        untuk memodelkan rezim fluktuasi harga gas (Sub-bab 3.2.1 & 3.5).
        """
        self.current_gas_gwei = float(gas_price_gwei)
        wei_hex = hex(int(round(self.current_gas_gwei * 1e9)))
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "anvil_setNextBlockBaseFeePerGas",
                "params": [wei_hex],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.rpc_client.rpc_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0):
                pass
        except Exception:
            pass

    @staticmethod
    def _parse_gas_used(tx_ok: bool, tx_info: str) -> int:
        if tx_info:
            m = _GAS_USED_RE.search(tx_info)
            if m:
                raw = m.group(1)
                try:
                    return int(raw, 16) if raw.startswith("0x") else int(raw)
                except ValueError:
                    pass
        return 45000 if tx_ok else 28000

    def execute_arm(self, arm: Arm) -> Tuple[bool, str]:
        target_addr = (
            self.vault_address
            if arm.target_contract == "VulnerableVault"
            else self.token_address
        )
        call_args: List[str] = []
        if arm.function_sig in ("emergencyDrain(address)", "emergencyWithdraw(address)"):
            call_args.append(self.rpc_client.agent_address)
        elif arm.function_sig == "mint(address,uint256)":
            call_args.extend([self.rpc_client.agent_address, str(int(250 * 1e18))])
        elif arm.function_sig == "transfer(address,uint256)":
            call_args.extend([self.vault_address, str(int(50 * 1e18))])

        # Snapshot target storage slots before executing the transaction
        pre_storage = self.rpc_client.snapshot_contract_storage(target_addr)

        tx_ok, tx_info = self.rpc_client.send_eoa_transaction(
            target_address=target_addr,
            function_sig=arm.function_sig,
            call_args=call_args,
            eth_value=arm.eth_value,
            gas_price_gwei=self.current_gas_gwei,
        )

        # Hybrid Greybox Opcode Trace + Public Storage Slot Diffing for x6 & x7
        tx_hash = self.rpc_client.extract_tx_hash(tx_info) if tx_ok else None
        sstore_val, waypoint_val = self.rpc_client.trace_transaction_dataflow(
            tx_hash=tx_hash,
            contract_address=target_addr,
            pre_storage=pre_storage,
        )
        self.last_sstore_flag = bool(sstore_val >= 0.5)
        self.last_waypoint_flag = bool(waypoint_val >= 0.5)

        return tx_ok, tx_info

    def evaluate_reward(
        self,
        prev_snap: OnChainContextSnapshot,
        next_snap: OnChainContextSnapshot,
        tx_succeeded: bool,
        arm_id: int,
    ) -> BanditRewardBreakdown:
        return self.reward_oracle.compute_reward(
            prev_snap=prev_snap,
            next_snap=next_snap,
            tx_succeeded=tx_succeeded,
            arm_id=arm_id,
        )

    def simulate_peer_user_deposit(self, eth_amount: float = 2.0) -> bool:
        """
        Simulate another normal user (Account #1) depositing ETH into the Vault.
        """
        ok, _ = self.rpc_client.send_eoa_transaction(
            target_address=self.vault_address,
            function_sig="deposit()",
            eth_value=eth_amount,
            sender_private_key=self.rpc_client.PEER_USER_PK,
            gas_price_gwei=self.current_gas_gwei,
        )
        return ok

    def simulate_peer_user_withdraw(self, eth_amount: float = 0.05) -> bool:
        """
        Simulasi pengguna pihak luar (Account #1) menarik sebagian simpanan sahnya dari Vault.
        """
        peer_stake = self.rpc_client.query_agent_stake(
            self.vault_address, self.rpc_client.PEER_USER_ADDR
        )
        if peer_stake < eth_amount:
            return self.simulate_peer_user_deposit(eth_amount=eth_amount)
        wei_amt = str(int(round(eth_amount * 1e18)))
        ok, _ = self.rpc_client.send_eoa_transaction(
            target_address=self.vault_address,
            function_sig="withdraw(uint256)",
            call_args=[wei_amt],
            sender_private_key=self.rpc_client.PEER_USER_PK,
            gas_price_gwei=self.current_gas_gwei,
        )
        return ok

    def simulate_external_party_event(self, round_idx: int) -> str:
        """
        Mengeksekusi aktivitas transaksi pihak luar pada putaran ke-t secara deterministik (CRN).
        """
        if not self.enable_peer_traffic:
            self.last_external_event = "none"
            return "none"
        idx = (max(1, round_idx) - 1) % len(self.external_events)
        ev = self.external_events[idx]
        ev_type = ev["type"]
        eth_amt = float(ev["eth_amount"])

        if ev_type == "peer_deposit":
            self.simulate_peer_user_deposit(eth_amount=eth_amt)
        elif ev_type == "peer_withdraw":
            self.simulate_peer_user_withdraw(eth_amount=eth_amt)
        elif ev_type == "peer_token_transfer":
            self.rpc_client.send_eoa_transaction(
                target_address=self.token_address,
                function_sig="mint(address,uint256)",
                call_args=[self.rpc_client.PEER_USER_ADDR, str(int(100 * 1e18))],
                sender_private_key=self.rpc_client.PEER_USER_PK,
                gas_price_gwei=self.current_gas_gwei,
            )
        self.last_external_event = ev_type
        return ev_type

    def prepare_round(self, round_idx: Optional[int] = None) -> None:
        """
        Menyiapkan harga gas blok (rezim gas) dan mengeksekusi transaksi pihak luar
        sebelum agen mengamati konteks x_t pada putaran ke-t.
        """
        t_idx = (round_idx if round_idx is not None else (self.round_counter + 1)) - 1
        g_t = self.gas_series[t_idx % len(self.gas_series)]
        self._set_network_base_fee(g_t)
        if self.enable_peer_traffic:
            self.simulate_external_party_event(t_idx + 1)

    def step(self, arm_id: int, prev_context: Optional[Tuple[np.ndarray, OnChainContextSnapshot]] = None) -> StepTransition:
        """
        Siklus langkah CMAB lengkap pada VaultCustodyEnv dengan perhitungan biaya c_t(a_t)
        (Persamaan 3.11) dan pembaruan sisa anggaran B_t (Persamaan 3.12).
        """
        if not (0 <= arm_id < len(self.arms)):
            raise ValueError(f"Invalid arm_id {arm_id}; expected 0 <= arm_id < {len(self.arms)}")

        self.round_counter += 1
        arm_spec = self.arms[arm_id]

        if prev_context is not None:
            prev_vec, prev_snap = prev_context
        else:
            prev_vec, prev_snap = self.observe_context()

        tx_ok, tx_info = self.execute_arm(arm_spec)
        self.recent_reverts.append(0 if tx_ok else 1)

        next_vec, next_snap = self.observe_context()
        reward_breakdown = self.evaluate_reward(
            prev_snap=prev_snap,
            next_snap=next_snap,
            tx_succeeded=tx_ok,
            arm_id=arm_id,
        )

        # Persamaan (3.11): c_t(a_t) = clip((GasUsed_t(a_t) * g_t) + v_{a_t}, c_min, c_max)
        gas_used = self._parse_gas_used(tx_ok, tx_info)
        g_t_eth = float(self.current_gas_gwei) * 1e-9
        v_at = float(arm_spec.eth_value)
        tx_cost = float(np.clip((gas_used * g_t_eth) + v_at, 0.001, 1.0))

        # Persamaan (3.12): B_t = max(0, B_{t-1} - c_t(a_t))
        self.remaining_budget_eth = max(0.0, self.remaining_budget_eth - tx_cost)

        return StepTransition(
            round_idx=self.round_counter,
            arm_spec=arm_spec,
            tx_succeeded=tx_ok,
            tx_hash_or_err=tx_info,
            prev_snapshot=prev_snap,
            next_snapshot=next_snap,
            context_vector=prev_vec,
            next_context_vector=next_vec,
            reward_breakdown=reward_breakdown,
            gas_price_gwei=self.current_gas_gwei,
            gas_used=gas_used,
            tx_cost_eth=tx_cost,
            remaining_budget_eth=self.remaining_budget_eth,
            external_event=self.last_external_event,
        )


class VaultCustodyCMABAgent:
    """
    High-level CMAB Agent wrapper for `Agen_Vault_Custody` supporting all 5 policies:
      - `d-lincbwk` / `d-linucbwk` (`DiscountedLinCBwK`)
      - `lincbwk` / `linucbwk` (`StationaryLinCBwK`)
      - `d-linucb` (`DiscountedLinUCB`)
      - `linucb` (`DisjointLinUCB`)
      - `uniform` / `random` (`UniformRandomPolicy`)
    """

    def __init__(
        self,
        env: VaultCustodyEnv,
        policy: Optional[BaseCMABPolicy] = None,
        algorithm: str = "d-lincbwk",
        alpha: float = 0.55,
        l2_lambda: float = 1.0,
        gamma: float = 0.82,
        initial_budget: Optional[float] = None,
        total_horizon: Optional[int] = None,
        use_theoretical_beta: bool = False,
        decay_all_arms: bool = False,
        seed: int = 42,
    ):
        self.env = env
        budget_b = float(initial_budget if initial_budget is not None else env.initial_budget_eth)
        horizon_t = int(total_horizon if total_horizon is not None else env.horizon_t)
        algo_key = algorithm.lower().strip()

        if policy is not None:
            self.policy = policy
        elif algo_key in ("d-lincbwk", "d_lincbwk", "d-linucbwk", "d_linucbwk", "discounted-lincbwk"):
            self.policy = DiscountedLinCBwK(
                n_arms=env.n_arms,
                context_dim=CONTEXT_DIM,
                gamma=gamma,
                initial_budget=budget_b,
                total_horizon=horizon_t,
                alpha=alpha,
                l2_lambda=l2_lambda,
                dual_update_mode="observed",
                decay_all_arms=decay_all_arms,
                seed=seed,
                name="D-LinCBwK",
            )
        elif algo_key in ("lincbwk", "linucbwk", "linucb-knapsack", "stationary-lincbwk"):
            self.policy = StationaryLinCBwK(
                n_arms=env.n_arms,
                context_dim=CONTEXT_DIM,
                initial_budget=budget_b,
                total_horizon=horizon_t,
                alpha=alpha,
                l2_lambda=l2_lambda,
                dual_update_mode="observed",
                seed=seed,
                name="LinCBwK",
            )
        elif algo_key in ("d-linucb", "discounted-linucb", "discounted_linucb"):
            self.policy = DiscountedLinUCB(
                n_arms=env.n_arms,
                context_dim=CONTEXT_DIM,
                gamma=gamma,
                alpha=alpha,
                l2_lambda=l2_lambda,
                use_theoretical_beta=use_theoretical_beta,
                decay_all_arms=decay_all_arms,
                seed=seed,
            )
        elif algo_key in ("uniform", "random", "uniform-random", "uniform_random"):
            self.policy = UniformRandomPolicy(
                n_arms=env.n_arms,
                context_dim=CONTEXT_DIM,
                initial_budget=budget_b,
                total_horizon=horizon_t,
                seed=seed,
                name="UniformRandom",
            )
        else:
            self.policy = DisjointLinUCB(
                n_arms=env.n_arms,
                context_dim=CONTEXT_DIM,
                alpha=alpha,
                l2_lambda=l2_lambda,
                gamma_forget=1.0,
                seed=seed,
            )

        self.discovered_vulnerabilities: List[Dict[str, Any]] = []

    def act_and_learn(self) -> Tuple[ArmSelectionResult, StepTransition]:
        """
        Execute one complete CMAB cycle on Anvil:
          1. Prepare round gas price g_t (gas regime) & execute external party tx (if enabled)
          2. Observe public context vector x_t in R^{10}
          3. Select arm a_t using active policy
          4. Broadcast unprivileged EOA transaction on Anvil & compute cost c_t(a_t)
          5. Update policy parameters with observed (reward r_t, cost c_t(a_t))
        """
        self.env.prepare_round(self.env.round_counter + 1)
        context_vec, prev_snap = self.env.observe_context()
        selection = self.policy.select_arm(context_vec)
        transition = self.env.step(selection.arm_id, prev_context=(context_vec, prev_snap))

        reward = transition.reward_breakdown.total_reward
        self.policy.update(
            arm_id=selection.arm_id,
            context=context_vec,
            reward=reward,
            cost=transition.tx_cost_eth,
        )

        for anomaly in transition.reward_breakdown.detected_anomalies:
            self.discovered_vulnerabilities.append(
                {
                    "round": transition.round_idx,
                    "arm_id": selection.arm_id,
                    "arm_name": transition.arm_spec.name,
                    "anomaly": anomaly,
                }
            )

        return selection, transition

    @property
    def feature_names(self) -> List[str]:
        return FEATURE_NAMES
