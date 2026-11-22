"""
Reward Function Modeling (R_Vault: X_Vault x A_Vault -> {0.00, 0.20, 0.50, 0.70, 1.00}) for Agen_Vault_Custody.

Implements the hierarchical 4-component state-transition reward structure (Persamaan 3.3 - 3.5):
    r_t(tau_t, I_t) = I(tau_t = 1) * min(1.00, w^T I_t)

Where event indicator vector I_t in {0, 1}^4 and hierarchical weight vector w = [0.20, 0.30, 0.20, 1.00]^T:
    - E_exec  (w_1 = 0.20) : Valid EVM execution without revert (tau_t = 1)
    - E_state (w_2 = 0.30) : Persistent state mutation in contract storage (Delta S_storage != 0)
    - E_econ  (w_3 = 0.20) : Persistent state mutation AND economic asset extraction to EOA
                             (Delta S_storage != 0 AND Delta Aset_eoa > eps)
    - E_vuln  (w_4 = 1.00) : Security invariant breach / dominant vulnerability override
                             (B_{t+1}^{(C)} < L_{t+1}^{(C)} OR Omega_t > Omega_{t+1})

Support set (Persamaan 3.5 & Tabel 3.2):
    supp(r_t) = V = {v_0, v_1, v_2, v_3, v_4} = {0.00, 0.20, 0.50, 0.70, 1.00}
    corresponding to mutually exclusive events E_0, E_1, E_2, E_3, E_4.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from .context import OnChainContextSnapshot

REWARD_SUPPORT_LEVELS = (0.00, 0.20, 0.50, 0.70, 1.00)
REWARD_WEIGHT_VECTOR = np.array([0.20, 0.30, 0.20, 1.00], dtype=np.float64)


@dataclass
class BanditRewardBreakdown:
    """
    Modular decomposition of the scalar reward r_t in {0.00, 0.20, 0.50, 0.70, 1.00}.
    """
    total_reward: float
    r_exec: float = 0.0
    r_state: float = 0.0
    r_econ: float = 0.0
    r_vuln: float = 0.0
    event_partition: str = "E0"
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
            "event_partition": self.event_partition,
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
    Evaluates the hierarchical 4-component scalar bandit reward r_t in {0.00, 0.20, 0.50, 0.70, 1.00}
    purely from public pre- and post-transaction EVM state transitions (Persamaan 3.3 - 3.5).
    """

    ASSET_GAIN_EPS: float = 0.001

    def compute_reward(
        self,
        prev_snap: OnChainContextSnapshot,
        next_snap: OnChainContextSnapshot,
        tx_succeeded: bool,
        arm_id: int = 0,
    ) -> BanditRewardBreakdown:
        # 1. Kejadian E0 (Level 0): Transaction Failed / Reverted in EVM (tau_t = 0) -> v_0 = 0.00
        if not tx_succeeded:
            return BanditRewardBreakdown(
                total_reward=0.00,
                r_exec=0.0,
                r_state=0.0,
                r_econ=0.0,
                r_vuln=0.0,
                event_partition="E0",
                detected_anomalies=[],
            )

        anomalies: List[str] = []

        # 2. Komponen 1: E_exec,t = I(tau_t = 1), w_1 = 0.20
        r_exec = 0.20

        # 3. Komponen 2: E_state,t = I(Delta S_storage,t != 0), w_2 = 0.30
        has_state_mutation = (
            getattr(next_snap, "sstore_executed", False)
            or abs(next_snap.agent_vault_balance - prev_snap.agent_vault_balance) > 1e-6
            or abs(next_snap.vault_total_deposited - prev_snap.vault_total_deposited) > 1e-6
        )
        r_state = 0.30 if has_state_mutation else 0.0

        # 4. Komponen 3: E_econ,t = I(Delta S_storage,t != 0 AND Delta Aset_eoa,t > eps), w_3 = 0.20
        delta_eoa_eth = next_snap.agent_eoa_eth_balance - prev_snap.agent_eoa_eth_balance
        delta_token = next_snap.agent_token_balance - prev_snap.agent_token_balance
        gained_assets = (delta_eoa_eth > self.ASSET_GAIN_EPS) or (delta_token > self.ASSET_GAIN_EPS)
        e_econ = bool(has_state_mutation and gained_assets)
        r_econ = 0.20 if e_econ else 0.0

        if e_econ and delta_eoa_eth > self.ASSET_GAIN_EPS:
            anomalies.append(f"ETH_EXTRACTION (+{delta_eoa_eth:.3f} ETH)")
        if e_econ and delta_token > self.ASSET_GAIN_EPS:
            anomalies.append(f"TOKEN_ACQUISITION (+{delta_token:.0f} Tokens)")

        # 5. Komponen 4: E_vuln,t = I((B_t >= L_t AND B_{t+1} < L_{t+1}) OR Omega_t > Omega_{t+1}), w_4 = 1.00
        solvency_breached = bool(prev_snap.vault_is_solvent and not next_snap.vault_is_solvent)
        contract_killed = bool(prev_snap.vault_code_alive and not next_snap.vault_code_alive)

        r_vuln = 0.0
        if solvency_breached:
            r_vuln = 1.00
            anomalies.append("SOLVENCY_INVARIANT_BREACH (vault_balance < totalDeposited)")
        if contract_killed:
            r_vuln = 1.00
            anomalies.append("CONTRACT_SELFDESTRUCTED (code destroyed)")

        # Persamaan (3.4): r_t = I(tau_t = 1) * min(1.00, w^T I_t)
        total = round(min(1.00, r_exec + r_state + r_econ + r_vuln), 2)

        # Partisi Kejadian E1..E4 (Tabel 3.2)
        if r_vuln >= 1.00:
            ev_part = "E4"
        elif e_econ:
            ev_part = "E3"
        elif has_state_mutation:
            ev_part = "E2"
        else:
            ev_part = "E1"

        return BanditRewardBreakdown(
            total_reward=total,
            r_exec=r_exec,
            r_state=r_state,
            r_econ=r_econ,
            r_vuln=r_vuln,
            event_partition=ev_part,
            detected_anomalies=anomalies,
        )
