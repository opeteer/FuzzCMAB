"""
Uniform Random Policy Baseline (Forge Fuzz Invariant Proxy).

Implements a context-free, uniform pseudo-random arm selector representing standard
Forge Fuzz / stateful invariant testing baselines for empirical benchmarking against
Contextual Multi-Armed Bandits (LinUCB, D-LinUCB, LinCBwK, and D-LinCBwK).
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base import BaseCMABPolicy, ArmSelectionResult


class UniformRandomPolicy(BaseCMABPolicy):
    """
    Uniform Random Policy representing Forge Fuzz / AFL-style uniform random mutator selection.

    At each discrete round t:
      1. Observes context vector x_t in R^d (recorded but ignored in decision).
      2. Selects an action a_t uniformly at random from candidate arms:
             P(a_t = a) = 1 / |Candidates|
      3. Receives bandit feedback (x_t, a_t, r_t, c_t).
      4. Tracks empirical statistics and cumulative costs without performing
         linear contextual regression or dual multiplier updates.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        initial_budget: float = 1.0,
        total_horizon: int = 24,
        seed: Optional[int] = 42,
        name: str = "UniformRandom",
    ):
        super().__init__(n_arms=n_arms, context_dim=context_dim, name=name)

        self.initial_budget = float(initial_budget)
        self.remaining_budget = float(initial_budget)
        self.total_horizon = int(total_horizon)
        self.rng = np.random.default_rng(seed)

        # Empirical tracking statistics
        self.cumulative_cost: float = 0.0
        self.empirical_sum_reward = np.zeros(n_arms, dtype=np.float64)
        self.empirical_sum_cost = np.zeros(n_arms, dtype=np.float64)

        # Compatibility attributes for benchmark runners and diagnostic telemetry
        self.theta = np.zeros((n_arms, context_dim), dtype=np.float64)
        self.psi = np.zeros((n_arms, context_dim), dtype=np.float64)
        self.A = np.zeros((n_arms, context_dim, context_dim), dtype=np.float64)
        self.A_inv = np.zeros((n_arms, context_dim, context_dim), dtype=np.float64)
        self.lambda_t: float = 0.0  # Zero dual penalty (unaware of gas knapsack constraints)

        self.dual_history: List[float] = [self.lambda_t]
        self.budget_history: List[float] = [self.remaining_budget]
        self.cost_history: List[float] = []
        self.reward_history: List[float] = []

        self.reset()

    def reset(self) -> None:
        """Reset all telemetry and counters to initial baseline state."""
        self.total_rounds = 0
        self.arm_counts.fill(0)
        self.cumulative_reward = 0.0
        self.cumulative_cost = 0.0
        self.remaining_budget = self.initial_budget
        self.empirical_sum_reward.fill(0.0)
        self.empirical_sum_cost.fill(0.0)

        self.theta.fill(0.0)
        self.psi.fill(0.0)
        eye = np.eye(self.context_dim, dtype=np.float64)
        for a in range(self.n_arms):
            self.A[a] = eye.copy()
            self.A_inv[a] = eye.copy()

        self.dual_history = [self.lambda_t]
        self.budget_history = [self.remaining_budget]
        self.cost_history.clear()
        self.reward_history.clear()

    def select_arm(
        self,
        context: np.ndarray,
        available_arms: Optional[List[int]] = None,
        filter_unaffordable: bool = False,
    ) -> ArmSelectionResult:
        """
        Select an arm uniformly at random:
            a_t ~ Uniform(Candidates)

        Parameters
        ----------
        context : np.ndarray
            Current context vector x_t in R^d (validated for dimension integrity).
        available_arms : Optional[List[int]]
            Subset of eligible candidate arms {0, ..., K-1}.
        filter_unaffordable : bool
            Retained for API compatibility with Knapsack policies.

        Returns
        -------
        ArmSelectionResult
            Result container with selected arm and uniform probability diagnostics.
        """
        _ = self.validate_context(context)

        candidates = (
            list(range(self.n_arms))
            if available_arms is None or len(available_arms) == 0
            else list(available_arms)
        )

        chosen_arm = int(self.rng.choice(candidates))

        prob = 1.0 / float(len(candidates))
        all_scores = {a: prob for a in candidates}

        # Calculate mean empirical reward of chosen arm (if pulled previously)
        n_pulls = self.arm_counts[chosen_arm]
        emp_mean = (
            float(self.empirical_sum_reward[chosen_arm] / max(1, n_pulls))
            if n_pulls > 0
            else 0.0
        )

        return ArmSelectionResult(
            arm_id=chosen_arm,
            ucb_score=prob,
            expected_reward=emp_mean,
            uncertainty_bonus=0.0,
            all_scores=all_scores,
            metadata={
                "algorithm": self.name,
                "strategy": "uniform_random",
                "prob": prob,
                "candidates": candidates,
                "remaining_budget": self.remaining_budget,
            },
        )

    def update(
        self,
        arm_id: int,
        context: np.ndarray,
        reward: float,
        cost: Optional[float] = None,
    ) -> None:
        """
        Update telemetry counters with bandit feedback (a_t, r_t, c_t).
        """
        if not (0 <= arm_id < self.n_arms):
            raise ValueError(f"Invalid arm_id {arm_id}; expected 0 <= arm_id < {self.n_arms}.")

        _ = self.validate_context(context)
        r = float(np.clip(reward, 0.0, 1.0))
        c = float(max(0.0, cost)) if cost is not None else 0.0

        self.total_rounds += 1
        self.arm_counts[arm_id] += 1
        self.cumulative_reward += r
        self.cumulative_cost += c
        self.remaining_budget = max(0.0, self.remaining_budget - c)

        self.empirical_sum_reward[arm_id] += r
        self.empirical_sum_cost[arm_id] += c

        # Maintain first component of theta as empirical mean for telemetry
        self.theta[arm_id, 0] = self.empirical_sum_reward[arm_id] / self.arm_counts[arm_id]
        if self.arm_counts[arm_id] > 0 and self.context_dim > 1:
            self.psi[arm_id, 0] = self.empirical_sum_cost[arm_id] / self.arm_counts[arm_id]

        self.dual_history.append(self.lambda_t)
        self.budget_history.append(self.remaining_budget)
        self.cost_history.append(c)
        self.reward_history.append(r)

    def update_with_cost(
        self,
        arm_id: int,
        context: np.ndarray,
        reward: float,
        cost: float,
    ) -> None:
        """Explicit semantic convenience method for CBwK feedback (x_t, a_t, r_t, c_t)."""
        self.update(arm_id=arm_id, context=context, reward=reward, cost=cost)

    def check_stopping_condition(
        self, context: Optional[np.ndarray] = None
    ) -> Tuple[bool, str]:
        """
        Stopping conditions:
          1. Horizon completed (t >= T)
          2. Budget exhausted (B_t <= 0)
        """
        if self.total_rounds >= self.total_horizon:
            return True, "horizon_completed"
        if self.remaining_budget <= 0.0:
            return True, "budget_exhausted"
        return False, "continue"

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic metrics dictionary."""
        return {
            "name": self.name,
            "total_rounds": self.total_rounds,
            "total_horizon": self.total_horizon,
            "arm_counts": self.arm_counts.tolist(),
            "cumulative_reward": self.cumulative_reward,
            "cumulative_cost": self.cumulative_cost,
            "remaining_budget": self.remaining_budget,
            "empirical_mean_rewards": [
                float(self.empirical_sum_reward[a] / max(1, self.arm_counts[a]))
                for a in range(self.n_arms)
            ],
            "empirical_mean_costs": [
                float(self.empirical_sum_cost[a] / max(1, self.arm_counts[a]))
                for a in range(self.n_arms)
            ],
        }

    def save_checkpoint(self, path: Path) -> None:
        """Serialize policy state to JSON."""
        state = {
            "name": self.name,
            "n_arms": self.n_arms,
            "context_dim": self.context_dim,
            "initial_budget": self.initial_budget,
            "remaining_budget": self.remaining_budget,
            "total_horizon": self.total_horizon,
            "total_rounds": self.total_rounds,
            "cumulative_reward": self.cumulative_reward,
            "cumulative_cost": self.cumulative_cost,
            "arm_counts": self.arm_counts.tolist(),
            "empirical_sum_reward": self.empirical_sum_reward.tolist(),
            "empirical_sum_cost": self.empirical_sum_cost.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_checkpoint(self, path: Path) -> None:
        """Load serialized policy state from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.total_rounds = int(state["total_rounds"])
        self.remaining_budget = float(state["remaining_budget"])
        self.cumulative_reward = float(state["cumulative_reward"])
        self.cumulative_cost = float(state["cumulative_cost"])
        self.arm_counts = np.array(state["arm_counts"], dtype=np.int64)
        self.empirical_sum_reward = np.array(state["empirical_sum_reward"], dtype=np.float64)
        self.empirical_sum_cost = np.array(state["empirical_sum_cost"], dtype=np.float64)

