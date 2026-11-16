"""
Reward Function Modeling (R_Vault: X_Vault x A_Vault -> [0.0, 1.0]) for Agen_Vault_Custody.

Implements a clean, human-interpretable, modular 4-component reward structure:
    r_t = 0.0 if REVERT
    r_t = min(1.0, R_exec + R_state + R_econ + R_vuln) if SUCCESS

Where:
    - R_exec  = 0.20 : Valid EVM execution without revert
    - R_state = 0.30 : Persistent state mutation in contract storage (SSTORE)
    - R_econ  = 0.20 : Economic asset gain (EOA ETH balance or Token balance increased)
    - R_vuln  = 0.30 : Security invariant breach (Solvency broken or Selfdestruct executed)

Zero hardcoded arm IDs or function name cheating: evaluated strictly from
public on-chain EVM state diffs (S_{t+1} - S_t).
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from .context import OnChainContextSnapshot


@dataclass
class BanditRewardBreakdown:
    """
    Modular decomposition of the scalar reward r_t in [0.0, 1.0].
    """
    total_reward: float
    r_exec: float = 0.0
    r_state: float = 0.0
    r_econ: float = 0.0
    r_vuln: float = 0.0
    detected_anomalies: List[str] = field(default_factory=list)
    revert_penalty_component: float = 0.0

    # Backwards-compatibility aliases for existing logging / analysis code
    @property
    def valid_execution_component(self) -> float:
        return self.r_exec

    @property
    def state_discovery_component(self) -> float:
        return self.r_state

    @property
    def net_eth_gain_component(self) -> float:
        return self.r_econ

    @property
    def vulnerability_signal_component(self) -> float:
        return self.r_vuln

    @property
    def token_utility_component(self) -> float:
        return 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_reward": self.total_reward,
            "r_exec": self.r_exec,
            "r_state": self.r_state,
            "r_econ": self.r_econ,
            "r_vuln": self.r_vuln,
            "valid_execution": self.r_exec,
            "state_discovery": self.r_state,
            "net_eth_gain": self.r_econ,
            "vulnerability_signal": self.r_vuln,
            "token_utility": 0.0,
            "revert_penalty": self.revert_penalty_component,
            "detected_anomalies": list(self.detected_anomalies),
        }


class VaultCustodyRewardOracle:
    """
    Evaluates the modular 4-component scalar bandit reward r_t in [0.0, 1.0]
    purely from public pre- and post-transaction EVM state transitions.
    """

    def compute_reward(
        self,
        prev_snap: OnChainContextSnapshot,
        next_snap: OnChainContextSnapshot,
        tx_succeeded: bool,
        arm_id: int = 0,
    ) -> BanditRewardBreakdown:
        # 1. Level 0: Transaction Failed / Reverted in EVM -> Zero Reward
        if not tx_succeeded:
            return BanditRewardBreakdown(
                total_reward=0.0,
                r_exec=0.0,
                r_state=0.0,
                r_econ=0.0,
                r_vuln=0.0,
                detected_anomalies=[],
            )

        anomalies: List[str] = []

        # 2. Level 1: R_exec (+0.20) - Valid on-chain execution
        r_exec = 0.20

        # 3. Level 2: R_state (+0.30) - Persistent storage mutation (SSTORE)
        has_state_mutation = (
            getattr(next_snap, "sstore_executed", False)
            or abs(next_snap.agent_vault_balance - prev_snap.agent_vault_balance) > 1e-6
            or abs(next_snap.vault_total_deposited - prev_snap.vault_total_deposited) > 1e-6
        )
        r_state = 0.30 if has_state_mutation else 0.0

        # 4. Level 3: R_econ (+0.20) - Economic asset gain (ETH extracted or Token gained)
        delta_eoa_eth = next_snap.agent_eoa_eth_balance - prev_snap.agent_eoa_eth_balance
        delta_token = next_snap.agent_token_balance - prev_snap.agent_token_balance
        gained_assets = (delta_eoa_eth > 0.001) or (delta_token > 0.001)
        r_econ = 0.20 if gained_assets else 0.0

        if delta_eoa_eth > 0.001:
            anomalies.append(f"ETH_EXTRACTION (+{delta_eoa_eth:.3f} ETH)")
        if delta_token > 0.001:
            anomalies.append(f"TOKEN_ACQUISITION (+{delta_token:.0f} Tokens)")

        # 5. Level 4: R_vuln (+0.30) - Security invariant / Bug oracle triggered
        solvency_breached = (
            prev_snap.vault_is_solvent and not next_snap.vault_is_solvent
        )
        contract_killed = (
            prev_snap.vault_code_alive and not next_snap.vault_code_alive
        )

        r_vuln = 0.0
        if solvency_breached:
            r_vuln = 0.30
            anomalies.append("SOLVENCY_INVARIANT_BREACH (vault_balance < totalDeposited)")
        if contract_killed:
            r_vuln = 0.30
            anomalies.append("CONTRACT_SELFDESTRUCTED (code destroyed)")

        # Sum of 4 discrete components bounded to [0.0, 1.0]
        total = round(min(1.0, r_exec + r_state + r_econ + r_vuln), 2)

        return BanditRewardBreakdown(
            total_reward=total,
            r_exec=r_exec,
            r_state=r_state,
            r_econ=r_econ,
            r_vuln=r_vuln,
            detected_anomalies=anomalies,
        )
