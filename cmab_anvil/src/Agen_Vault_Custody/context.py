"""
Context Space Modeling (X_Vault subset R^10) for Agen_Vault_Custody (Refactored 10D Optimized).

Implements a scale-invariant, magic-number-free, and CBwK-compliant 10-dimensional
context vector x_t in R^{10} combining relative on-chain financial ratios with
Hybrid Greybox/Black-box EVM execution trace signals (SSTORE & SLOAD->SSTORE waypoints).
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

CONTEXT_DIM: int = 10

FEATURE_NAMES: List[str] = [
    "x0_bias",                     # Constant intercept = 1.0
    "x1_vault_liquidity_ratio",    # Vault_ETH / (Vault_ETH + Agent_ETH + 1e-5) in [0.0, 1.0]
    "x2_claim_liability_ratio",    # Total_Deposited / (Vault_ETH + Total_Deposited + 1e-5) in [0.0, 1.0]
    "x3_solvency_gap_ratio",       # clip((Vault_ETH - Total_Deposited) / max(Total_Deposited, 1.0), -1.0, 1.0)
    "x4_agent_stake_ratio",        # clip(Agent_Vault_Stake / (Total_Deposited + 1e-5), 0.0, 1.0)
    "x5_deposit_eligibility",      # min(1.0, Agent_EOA_ETH / Min_Deposit_Requirement) in [0.0, 1.0]
    "x6_sstore_execution_flag",    # 1.0 if last transaction triggered SSTORE (storage write), else 0.0
    "x7_dataflow_waypoint_flag",   # 1.0 if SSTORE modified a slot previously read via SLOAD, else 0.0
    "x8_recent_revert_rate",       # Moving average revert rate of agent's recent transactions in [0.0, 1.0]
    "x9_vault_code_alive",         # 1.0 if target contract bytecode is intact, 0.0 if selfdestructed
]


@dataclass
class OnChainContextSnapshot:
    """
    On-chain state and EVM execution trace snapshot observed by Agen_Vault_Custody at round t.
    """
    block_number: int
    vault_eth_balance: float
    vault_total_deposited: float
    agent_vault_balance: float
    agent_eoa_eth_balance: float
    agent_token_balance: float
    vault_is_solvent: bool
    vault_is_locked: bool
    vault_code_alive: bool
    recent_revert_rate: float
    sstore_executed: bool = False
    dataflow_waypoint: bool = False
    min_deposit_requirement: float = 0.25


class PublicContextExtractor:
    """
    Maps an `OnChainContextSnapshot` into a scale-invariant, collinearity-free
    10-dimensional context vector x_t in R^{10} without hardcoded magic numbers.
    """

    EPS: float = 1e-5

    def __init__(
        self,
        min_deposit_eth: float = 0.25,
        vault_cap_eth: Optional[float] = None,
        agent_init_eth: Optional[float] = None,
    ):
        # `vault_cap_eth` and `agent_init_eth` retained as optional kwargs for backward compatibility
        self.min_deposit_eth = max(min_deposit_eth, self.EPS)

    def to_vector(self, snap: OnChainContextSnapshot) -> np.ndarray:
        vault_eth = max(0.0, float(snap.vault_eth_balance))
        total_dep = max(0.0, float(snap.vault_total_deposited))
        agent_stake = max(0.0, float(snap.agent_vault_balance))
        agent_eth = max(0.0, float(snap.agent_eoa_eth_balance))

        # x0: Constant intercept term for linear UCB regression
        x0_bias = 1.0

        # x1: Relative Vault Liquidity Ratio (scale-free in [0.0, 1.0])
        x1_vault_liq_ratio = float(
            np.clip(vault_eth / (vault_eth + agent_eth + self.EPS), 0.0, 1.0)
        )

        # x2: Relative Claim Liability Ratio (scale-free in [0.0, 1.0])
        x2_liability_ratio = float(
            np.clip(total_dep / (vault_eth + total_dep + self.EPS), 0.0, 1.0)
        )

        # x3: Solvency Gap Ratio in [-1.0, 1.0] (negative indicates insolvency / drain)
        denom_solvency = max(total_dep, 1.0)
        x3_solvency_gap = float(
            np.clip((vault_eth - total_dep) / denom_solvency, -1.0, 1.0)
        )

        # x4: Agent's proportional stake share inside the Vault in [0.0, 1.0]
        # Fallback: if total_dep is not tracked by the contract (0.0) but agent_stake > 0,
        # normalize against (vault_eth + agent_stake + EPS).
        stake_denom = (
            (total_dep + self.EPS)
            if total_dep >= agent_stake and total_dep > 0.0
            else (max(vault_eth, agent_stake) + self.EPS)
        )
        x4_agent_stake_ratio = float(np.clip(agent_stake / stake_denom, 0.0, 1.0))

        # x5: Instant Deposit Eligibility Ratio in [0.0, 1.0] (CBwK-compliant feasibility signal,
        # decoupled from global gas budget trajectory)
        req_dep = max(
            float(getattr(snap, "min_deposit_requirement", self.min_deposit_eth)),
            self.EPS,
        )
        x5_deposit_eligibility = float(np.clip(min(1.0, agent_eth / req_dep), 0.0, 1.0))

        # x6: EVM SSTORE Execution Flag (1.0 if last tx wrote persistent storage, else 0.0)
        x6_sstore_flag = 1.0 if getattr(snap, "sstore_executed", False) else 0.0

        # x7: SMARTIAN / ItyFuzz Dataflow Waypoint Flag (1.0 if SSTORE modified a previously SLOADed slot)
        x7_waypoint_flag = 1.0 if getattr(snap, "dataflow_waypoint", False) else 0.0

        # x8: Moving-average transaction revert rate in [0.0, 1.0]
        x8_revert_rate = float(np.clip(snap.recent_revert_rate, 0.0, 1.0))

        # x9: Contract bytecode liveness indicator (0.0 if selfdestructed)
        x9_code_alive = 1.0 if snap.vault_code_alive else 0.0

        return np.array(
            [
                x0_bias,
                x1_vault_liq_ratio,
                x2_liability_ratio,
                x3_solvency_gap,
                x4_agent_stake_ratio,
                x5_deposit_eligibility,
                x6_sstore_flag,
                x7_waypoint_flag,
                x8_revert_rate,
                x9_code_alive,
            ],
            dtype=np.float64,
        )
