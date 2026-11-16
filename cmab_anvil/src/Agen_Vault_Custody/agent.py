"""
Agen_Vault_Custody: Environment & Agent Orchestrator for Custody Vault & ERC-20 Contracts.

Connects the domain-specific Vault/Token modeling (`context.py`, `arms.py`, `reward.py`)
with the universal CMAB strategy templates (`cmab_anvil/algorithms/linucb.py` and
`cmab_anvil/algorithms/discounted_linucb.py`) and universal Anvil RPC client
(`cmab_anvil/common/anvil_rpc.py`).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..algorithms.base import BaseCMABPolicy, ArmSelectionResult
from ..algorithms.linucb import DisjointLinUCB
from ..algorithms.discounted_linucb import DiscountedLinUCB
from ..common.anvil_rpc import UnprivilegedAnvilClient
from ..common.base_env import BaseArmSpec, BaseUnprivilegedBanditEnv, StepTransition

from .context import (
    OnChainContextSnapshot,
    PublicContextExtractor,
    CONTEXT_DIM,
    FEATURE_NAMES,
)
from .arms import build_vault_custody_arm_catalog
from .reward import BanditRewardBreakdown, VaultCustodyRewardOracle


class VaultCustodyEnv(BaseUnprivilegedBanditEnv):
    """
    Concrete CMAB environment for `Agen_Vault_Custody` interacting with
    `VulnerableVault` and `MockToken` on Anvil without any special privileges.
    """

    def __init__(
        self,
        rpc_url: str = "http://127.0.0.1:8545",
        deployed_json_path: Optional[Path] = None,
        agent_ref_budget_eth: float = 5.0,
    ):
        rpc_client = UnprivilegedAnvilClient(rpc_url=rpc_url)
        arms = build_vault_custody_arm_catalog()
        super().__init__(rpc_client=rpc_client, arms=arms, revert_window=10)

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

    def execute_arm(self, arm: BaseArmSpec) -> Tuple[bool, str]:
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
        )
        return ok


class VaultCustodyCMABAgent:
    """
    High-level CMAB Agent wrapper for `Agen_Vault_Custody` driven by universal
    CMAB strategy templates (`DisjointLinUCB` or `DiscountedLinUCB`).
    """

    def __init__(
        self,
        env: VaultCustodyEnv,
        policy: Optional[BaseCMABPolicy] = None,
        algorithm: str = "linucb",
        alpha: float = 1.25,
        l2_lambda: float = 1.0,
        gamma: float = 0.85,
        use_theoretical_beta: bool = True,
        decay_all_arms: bool = True,
        seed: int = 42,
    ):
        self.env = env
        if policy is not None:
            self.policy = policy
        elif algorithm.lower() in ("d-linucb", "discounted-linucb", "discounted_linucb"):
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
          1. Observe public context vector x_t in R^{10}
          2. Select arm a_t using active policy (LinUCB or D-LinUCB)
          3. Broadcast unprivileged EOA transaction on Anvil
          4. Update policy parameters with observed reward r_t
        """
        context_vec, _ = self.env.observe_context()
        selection = self.policy.select_arm(context_vec)
        transition = self.env.step(selection.arm_id)

        reward = transition.reward_breakdown.total_reward
        self.policy.update(
            arm_id=selection.arm_id,
            context=context_vec,
            reward=reward,
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
