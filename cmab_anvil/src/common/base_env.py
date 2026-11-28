"""
Universal Base Environment and Data Structures for Regular User CMAB Agents (Agen Pengguna Biasa) on Anvil.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Tuple
import numpy as np

from .anvil_rpc import RegularUserAnvilClient


@dataclass(frozen=True)

class Arm:
    arm_id: int
    name: str
    target_contract: str
    function_sig: str
    eth_value: float
    description: str


@dataclass
class StepTransition:
    """
    Universal record of a single CMAB interaction round t on Anvil.
    """
    round_idx: int
    arm_spec: Arm
    tx_succeeded: bool
    tx_hash_or_err: str
    prev_snapshot: Any
    next_snapshot: Any
    context_vector: np.ndarray
    next_context_vector: np.ndarray
    reward_breakdown: Any
    gas_price_gwei: float = 1.0
    gas_used: int = 21000
    tx_cost_eth: float = 0.001
    remaining_budget_eth: float = 5.0
    external_event: str = "none"


class BaseRegularUserBanditEnv(ABC):
    """
    Abstract base class for regular user CMAB environments (agen pengguna biasa) on Anvil.
    Provides shared RPC client management, revert rate tracking, and step loop structure.
    """

    def __init__(
        self,
        rpc_client: RegularUserAnvilClient,
        arms: List[Arm],
        revert_window: int = 10,
    ):
        self.rpc_client = rpc_client
        self.arms = arms
        self.recent_reverts: Deque[int] = deque(maxlen=revert_window)
        self.round_counter: int = 0

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    @property
    def recent_revert_rate(self) -> float:
        if len(self.recent_reverts) == 0:
            return 0.0
        return float(sum(self.recent_reverts)) / len(self.recent_reverts)

    @abstractmethod
    def verify_regular_user_standing(self) -> Dict[str, str]:
        """Verify that the agent acts strictly as a regular user (pengguna biasa) on target contracts."""
        pass

    @abstractmethod
    def observe_context(self) -> Tuple[np.ndarray, Any]:
        """Observe public on-chain state and return (context_vector, raw_snapshot)."""
        pass

    @abstractmethod
    def execute_arm(self, arm: Arm) -> Tuple[bool, str]:
        """Execute the chosen EOA transaction arm on Anvil."""
        pass

    @abstractmethod
    def evaluate_reward(
        self,
        prev_snap: Any,
        next_snap: Any,
        tx_succeeded: bool,
        arm_id: int,
    ) -> Any:
        """Compute bandit reward breakdown from pre- and post-transaction public snapshots."""
        pass

    def step(self, arm_id: int) -> StepTransition:
        """
        Universal CMAB step lifecycle:
          1. Observe pre-transaction context x_t
          2. Execute EOA transaction for arm a_t
          3. Observe post-transaction state
          4. Compute scalar reward r_t and breakdown
        """
        if not (0 <= arm_id < len(self.arms)):
            raise ValueError(f"Invalid arm_id {arm_id}; expected 0 <= arm_id < {len(self.arms)}")

        self.round_counter += 1
        arm_spec = self.arms[arm_id]

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
        )

