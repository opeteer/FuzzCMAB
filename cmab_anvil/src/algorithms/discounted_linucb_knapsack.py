"""
Discounted Contextual Multi-Armed Bandits with Knapsacks (D-LinCBwK).

Mathematical Formulation:
    This algorithm marries two foundational frameworks:
      1. Discounted LinUCB (D-LinUCB): Handles non-stationary EVM state shifts,
         transient liquidity dynamics, and protocol state transitions by discounting
         historical observations with decay factor gamma in (0, 1).
      2. Lagrangian Dual Knapsack (CBwK): Controls the agent's global capital/gas budget
         via an adaptive shadow price (dual multiplier lambda_t) entirely outside the
         context vector x_t, preserving scale-invariance and preventing regression bias.

1. Definitions & Parameter Spaces:
    - t in {1, 2, ..., T}       : Transaction / round index.
    - a in A = {0, ..., K-1}    : Set of K discrete mutator actions (arms).
    - x_t in R^d (||x_t||_2<=1) : Pure EVM state context vector (excluding remaining budget).
    - r_{t,a} in [0, 1]          : Realized bandit reward (coverage gain / security finding).
    - c_{t,a} in (0, 1]          : Realized normalized transaction gas cost.
    - B_t in R_+ (B_1 = B)      : Remaining global capital/gas budget at round t.
    - gamma in (0, 1)           : Exponential memory discount factor.
    - lambda_t >= 0             : Lagrangian dual multiplier (shadow price of gas).
    - alpha > 0                 : Exploration hyperparameter for reward UCB bonus.
    - beta >= 0                 : Exploration hyperparameter for cost LCB (or 0 for expectation).

2. Core Mathematical Stages:
    Stage A (Discounted LinUCB Reward Estimator):
        A_{t,a,gamma} = sum_{s=1}^{t-1} gamma^{t-s} I(a_s = a) x_s x_s^T + lambda_{ridge} * I_d
        b_{t,a,gamma} = sum_{s=1}^{t-1} gamma^{t-s} I(a_s = a) r_s x_s
        hat{theta}_{t,a,gamma} = A_{t,a,gamma}^{-1} b_{t,a,gamma}
        hat{r}_{t,a}(x_t) = hat{theta}_{t,a,gamma}^T x_t + alpha * sqrt(x_t^T A_{t,a,gamma}^{-1} x_t)

    Stage B (Discounted Cost Estimator):
        b_{t,a,gamma}^{(c)} = sum_{s=1}^{t-1} gamma^{t-s} I(a_s = a) c_s x_s
        hat{psi}_{t,a,gamma} = A_{t,a,gamma}^{-1} b_{t,a,gamma}^{(c)}
        hat{c}_{t,a}(x_t) = max( min_cost, hat{psi}_{t,a,gamma}^T x_t )

    Stage C (Global Knapsack & Dual Multiplier Update):
        B_t = B_{t-1} - c_{t-1, a_{t-1}}
        Average quota per remaining round: q_t = B_t / (T - t + 1)
        Target consumption rate: rho = B / T
        eta_t = sqrt( ln(K) / T )
        lambda_t = lambda_{t-1} * exp( eta_t * ( hat{c}_{t-1, a_{t-1}}(x_{t-1}) - B / T ) )

    Stage D (Final Decision Rule):
        a_t = argmax_{a in A} [ hat{r}_{t,a}(x_t) - lambda_t * hat{c}_{t,a}(x_t) ]

3. Recursive Update Algorithm (Sherman-Morrison Formula O(d^2)):
        A_{t,a,gamma}^{-1} = (1 / gamma) * A_{t-1,a,gamma}^{-1}
          - [ (1 / gamma^2) * A_{t-1,a,gamma}^{-1} x_t x_t^T A_{t-1,a,gamma}^{-1} ]
            / [ 1 + (1 / gamma) * x_t^T A_{t-1,a,gamma}^{-1} x_t ]
        b_{t,a,gamma} = gamma * b_{t-1,a,gamma} + r_t * x_t
        b_{t,a,gamma}^{(c)} = gamma * b_{t-1,a,gamma}^{(c)} + c_t * x_t

4. Dual Stopping Horizons:
        Stopping occurs if:
          - t >= T (Horizon completed)
          - B_t <= 0 (Budget exhausted)
          - B_t < min_{a in A} hat{c}_{t,a}(x_t) (Insufficient capital for any arm)
"""

from typing import Dict, List, Optional, Tuple, Any
import json
import math
from pathlib import Path
import numpy as np

from .base import BaseCMABPolicy, ArmSelectionResult


class DiscountedLinCBwK(BaseCMABPolicy):
    """
    Discounted Contextual Multi-Armed Bandits with Knapsacks (D-LinCBwK).

    Implements dual-learning for non-stationary contextual optimization under
    global Knapsack resource constraints.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        initial_budget: float = 1.0,
        total_horizon: int = 24,
        gamma: float = 0.82,
        alpha: float = 0.15,
        beta: float = 0.0,
        l2_lambda: float = 1.0,
        initial_lambda: float = 1.0,
        min_cost: float = 0.001,
        use_sherman_morrison: bool = True,
        decay_all_arms: bool = True,
        dual_update_mode: str = "estimated",
        lambda_min: float = 1e-4,
        lambda_max: float = 1e4,
        custom_eta: Optional[float] = None,
        seed: Optional[int] = 42,
        name: str = "DiscountedLinCBwK",
    ):
        """
        Parameters
        ----------
        n_arms : int
            Number of discrete actions K (mutators / transactions).
        context_dim : int
            Dimension d of context vector x_t in R^d (pure EVM state).
        initial_budget : float
            Total initial capital / gas budget B > 0 (e.g., 0.50 ETH).
        total_horizon : int
            Planned total interaction rounds T >= 1.
        gamma : float
            Memory discount factor gamma in (0.0, 1.0]. Typical range [0.75, 0.95].
        alpha : float
            Upper Confidence Bound exploration parameter for reward estimation (alpha > 0).
        beta : float
            Lower Confidence Bound parameter for cost estimation (beta >= 0).
            If 0.0, uses expected cost directly.
        l2_lambda : float
            Ridge regression L2 regularization parameter lambda_ridge > 0.
        initial_lambda : float
            Initial value for Lagrangian dual shadow price lambda_1 > 0.
        min_cost : float
            Strict lower bound on cost prediction to avoid zero/negative denominators.
        use_sherman_morrison : bool
            If True, uses O(d^2) rank-1 Sherman-Morrison recursive formula.
        decay_all_arms : bool
            If True, discounts all arms synchronously at every round.
        dual_update_mode : str
            "estimated" (uses hat{c}_{t-1}(x_{t-1})) or "observed" (uses c_{t-1}).
        lambda_min : float
            Strict lower bound for dual multiplier lambda_t.
        lambda_max : float
            Strict upper bound for dual multiplier lambda_t to avoid numerical overflow.
        custom_eta : Optional[float]
            Optional manual learning rate for dual update. If None, uses sqrt(ln(K) / T).
        seed : Optional[int]
            Random seed for deterministic tie-breaking.
        name : str
            Identifier string for telemetry and logging.
        """
        super().__init__(n_arms=n_arms, context_dim=context_dim, name=name)

        if initial_budget <= 0.0:
            raise ValueError("initial_budget must be strictly positive.")
        if total_horizon <= 0:
            raise ValueError("total_horizon must be strictly positive.")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0.0, 1.0].")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative.")
        if beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if l2_lambda <= 0.0:
            raise ValueError("l2_lambda must be strictly positive.")
        if initial_lambda <= 0.0:
            raise ValueError("initial_lambda must be strictly positive for exponential updates.")
        if min_cost <= 0.0:
            raise ValueError("min_cost must be strictly positive.")

        self.initial_budget = float(initial_budget)
        self.remaining_budget = float(initial_budget)
        self.total_horizon = int(total_horizon)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.l2_lambda = float(l2_lambda)
        self.initial_lambda = float(initial_lambda)
        self.lambda_t = float(initial_lambda)
        self.min_cost = float(min_cost)
        self.use_sherman_morrison = bool(use_sherman_morrison)
        self.decay_all_arms = bool(decay_all_arms)
        self.dual_update_mode = str(dual_update_mode)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.custom_eta = float(custom_eta) if custom_eta is not None else None
        self.rng = np.random.default_rng(seed)

        # Dual learning rate eta_t = sqrt(ln(K) / T)
        self.eta = self.custom_eta if self.custom_eta is not None else math.sqrt(math.log(max(2, n_arms)) / total_horizon)

        # Stage A & B Matrices and Vectors
        # Covariance matrix A_a and inverse A_inv_a for each arm
        self.A = np.zeros((n_arms, context_dim, context_dim), dtype=np.float64)
        self.A_inv = np.zeros((n_arms, context_dim, context_dim), dtype=np.float64)

        # Reward response vector b_a and parameter theta_a
        self.b_reward = np.zeros((n_arms, context_dim), dtype=np.float64)
        self.theta = np.zeros((n_arms, context_dim), dtype=np.float64)

        # Cost response vector b_c_a and parameter psi_a
        self.b_cost = np.zeros((n_arms, context_dim), dtype=np.float64)
        self.psi = np.zeros((n_arms, context_dim), dtype=np.float64)

        # Tracking state
        self.last_context: Optional[np.ndarray] = None
        self.last_chosen_arm: Optional[int] = None
        self.last_predicted_cost: float = self.min_cost
        self.cumulative_cost: float = 0.0

        # Telemetry histories
        self.dual_history: List[float] = [self.lambda_t]
        self.budget_history: List[float] = [self.remaining_budget]
        self.cost_history: List[float] = []
        self.reward_history: List[float] = []

        self.reset()

    def reset(self) -> None:
        """Reset all learned matrices, vectors, budget, and dual price to initial priors."""
        self.total_rounds = 0
        self.arm_counts.fill(0)
        self.cumulative_reward = 0.0
        self.cumulative_cost = 0.0
        self.remaining_budget = self.initial_budget
        self.lambda_t = self.initial_lambda

        self.last_context = None
        self.last_chosen_arm = None
        self.last_predicted_cost = self.min_cost

        self.dual_history = [self.lambda_t]
        self.budget_history = [self.remaining_budget]
        self.cost_history.clear()
        self.reward_history.clear()

        eye = np.eye(self.context_dim, dtype=np.float64)
        for a in range(self.n_arms):
            self.A[a] = self.l2_lambda * eye
            self.A_inv[a] = (1.0 / self.l2_lambda) * eye
            self.b_reward[a].fill(0.0)
            self.theta[a].fill(0.0)
            self.b_cost[a].fill(0.0)
            self.psi[a].fill(0.0)

    # -------------------------------------------------------------------------
    # Estimators: Reward & Gas Cost
    # -------------------------------------------------------------------------
    def estimate_reward(self, arm_id: int, x: np.ndarray) -> Tuple[float, float, float]:
        """
        Stage A: Compute Discounted LinUCB Reward Estimator.
            hat{r}_{t,a}(x_t) = hat{theta}_{t,a,gamma}^T x_t + alpha * sqrt(x_t^T A_{t,a,gamma}^{-1} x_t)

        Returns
        -------
        Tuple[float, float, float]
            (expected_mean, uncertainty_bonus, ucb_score)
        """
        x_col = self.validate_context(x)
        mean_est = float(np.dot(self.theta[arm_id], x_col))

        # Quadratic Mahalanobis uncertainty form: x^T @ A^{-1} @ x
        quad = float(np.dot(x_col, np.dot(self.A_inv[arm_id], x_col)))
        uncertainty = self.alpha * math.sqrt(max(0.0, quad))
        ucb = mean_est + uncertainty
        return mean_est, uncertainty, ucb

    def estimate_cost(self, arm_id: int, x: np.ndarray) -> Tuple[float, float]:
        """
        Stage B: Compute Discounted Gas Cost Estimator.
            hat{c}_{t,a}(x_t) = max( min_cost, hat{psi}_{t,a,gamma}^T x_t )

        Returns
        -------
        Tuple[float, float]
            (expected_cost, bounded_predicted_cost)
        """
        x_col = self.validate_context(x)
        expected_cost = float(np.dot(self.psi[arm_id], x_col))

        if self.beta > 0.0:
            quad = float(np.dot(x_col, np.dot(self.A_inv[arm_id], x_col)))
            lcb_bonus = self.beta * math.sqrt(max(0.0, quad))
            predicted_cost = max(self.min_cost, expected_cost - lcb_bonus)
        else:
            predicted_cost = max(self.min_cost, expected_cost)

        return expected_cost, predicted_cost

    # -------------------------------------------------------------------------
    # Decision Rule (Stage D)
    # -------------------------------------------------------------------------
    def select_arm(
        self,
        context: np.ndarray,
        available_arms: Optional[List[int]] = None,
        filter_unaffordable: bool = True,
    ) -> ArmSelectionResult:
        """
        Stage D: Select optimal arm using Lagrangian Dual Decision Rule:
            a_t = argmax_{a in A} [ hat{r}_{t,a}(x_t) - lambda_t * hat{c}_{t,a}(x_t) ]

        Parameters
        ----------
        context : np.ndarray
            Current pure EVM status context vector x_t in R^d.
        available_arms : Optional[List[int]]
            Candidate arm subset. Defaults to all arms {0, ..., K-1}.
        filter_unaffordable : bool
            If True, removes arms whose predicted cost exceeds remaining budget B_t,
            provided at least one affordable arm exists.
        """
        x = self.validate_context(context)
        candidates = (
            list(range(self.n_arms))
            if available_arms is None or len(available_arms) == 0
            else list(available_arms)
        )

        all_scores: Dict[int, float] = {}
        expected_rewards: Dict[int, float] = {}
        uncertainties: Dict[int, float] = {}
        ucb_rewards: Dict[int, float] = {}
        predicted_costs: Dict[int, float] = {}
        lagrangian_penalties: Dict[int, float] = {}

        for a in candidates:
            mean_r, unc_r, ucb_r = self.estimate_reward(a, x)
            _, pred_c = self.estimate_cost(a, x)

            penalty = self.lambda_t * pred_c
            lagrangian_score = ucb_r - penalty

            all_scores[a] = lagrangian_score
            expected_rewards[a] = mean_r
            uncertainties[a] = unc_r
            ucb_rewards[a] = ucb_r
            predicted_costs[a] = pred_c
            lagrangian_penalties[a] = penalty

        # Optional Knapsack feasibility filter
        if filter_unaffordable and self.remaining_budget > 0.0:
            affordable = [a for a in candidates if predicted_costs[a] <= self.remaining_budget]
            if affordable:
                candidates = affordable

        # Argmax with deterministic tie-breaking
        best_score = -np.inf
        best_arms: List[int] = []

        for a in candidates:
            score = all_scores[a]
            if score > best_score + 1e-12:
                best_score = score
                best_arms = [a]
            elif abs(score - best_score) <= 1e-12:
                best_arms.append(a)

        chosen_arm = int(self.rng.choice(best_arms)) if best_arms else candidates[0]

        # Record state for next dual update
        self.last_context = x.copy()
        self.last_chosen_arm = chosen_arm
        self.last_predicted_cost = predicted_costs[chosen_arm]

        # Calculate remaining quota per round
        rounds_remaining = max(1, self.total_horizon - self.total_rounds)
        quota_per_round = self.remaining_budget / float(rounds_remaining)

        return ArmSelectionResult(
            arm_id=chosen_arm,
            ucb_score=all_scores[chosen_arm],
            expected_reward=expected_rewards[chosen_arm],
            uncertainty_bonus=uncertainties[chosen_arm],
            all_scores=all_scores,
            metadata={
                "algorithm": self.name,
                "gamma": self.gamma,
                "lambda_dual": self.lambda_t,
                "remaining_budget": self.remaining_budget,
                "quota_per_round": quota_per_round,
                "chosen_predicted_cost": predicted_costs[chosen_arm],
                "chosen_ucb_reward": ucb_rewards[chosen_arm],
                "chosen_lagrangian_penalty": lagrangian_penalties[chosen_arm],
                "expected_rewards": expected_rewards,
                "ucb_rewards": ucb_rewards,
                "predicted_costs": predicted_costs,
                "lagrangian_penalties": lagrangian_penalties,
            },
        )

    # -------------------------------------------------------------------------
    # Recursive Updates (Stage C & Section 3)
    # -------------------------------------------------------------------------
    def _sherman_morrison_rank1(
        self, A_inv: np.ndarray, x: np.ndarray, gamma: float
    ) -> np.ndarray:
        """
        Section 3: Recursive Matrix Inversion Update via Sherman-Morrison Formula:
            A_{t,a,gamma}^{-1} = (1 / gamma) * A_{t-1,a,gamma}^{-1}
              - [ (1 / gamma^2) * A_{t-1,a,gamma}^{-1} x_t x_t^T A_{t-1,a,gamma}^{-1} ]
                / [ 1 + (1 / gamma) * x_t^T A_{t-1,a,gamma}^{-1} x_t ]

        Complexity: O(d^2) operations, strictly avoiding O(d^3) inversion.
        """
        # Let u = (1 / gamma) * A_inv @ x
        u = (1.0 / gamma) * np.dot(A_inv, x)
        denom = 1.0 + float(np.dot(x, u))

        # (1 / gamma^2) * A_inv @ x @ x^T @ A_inv = u @ u^T
        outer_u = np.outer(u, u)
        A_inv_new = (1.0 / gamma) * A_inv - (outer_u / max(1e-12, denom))

        # Numerical symmetrization
        return 0.5 * (A_inv_new + A_inv_new.T)

    def update(
        self,
        arm_id: int,
        context: np.ndarray,
        reward: float,
        cost: Optional[float] = None,
    ) -> None:
        """
        Execute recursive update for Stage A (reward), Stage B (cost), and Stage C (dual Knapsack).

        Parameters
        ----------
        arm_id : int
            Pulled arm index a_t in {0, ..., K-1}.
        context : np.ndarray
            EVM state context vector x_t in R^d.
        reward : float
            Realized coverage / waypoint reward r_t in [0.0, 1.0].
        cost : Optional[float]
            Realized transaction gas cost c_t > 0.
            If None, defaults to the arm's last predicted cost.
        """
        if not (0 <= arm_id < self.n_arms):
            raise ValueError(f"Invalid arm_id {arm_id}; expected 0 <= arm_id < {self.n_arms}.")

        x = self.validate_context(context)
        r = float(np.clip(reward, 0.0, 1.0))
        c = (
            float(np.clip(cost, self.min_cost, 10.0))
            if cost is not None
            else self.last_predicted_cost
        )

        eye = np.eye(self.context_dim, dtype=np.float64)

        # ---------------------------------------------------------------------
        # 1. Stage C: Lagrangian Dual Multiplier & Budget Decay
        # ---------------------------------------------------------------------
        self.remaining_budget = max(0.0, self.remaining_budget - c)

        # Target consumption rate per round: rho = B / T
        target_rate = self.initial_budget / float(self.total_horizon)

        # Determine cost metric for dual gradient update
        dual_cost_signal = self.last_predicted_cost if self.dual_update_mode == "estimated" else c
        dual_gradient = dual_cost_signal - target_rate

        # Exponential Dual Update: lambda_t = lambda_{t-1} * exp( eta_t * (cost - B/T) )
        step = self.eta * dual_gradient
        # Clip step to [-5.0, 5.0] to prevent math.exp overflow
        clipped_step = float(np.clip(step, -5.0, 5.0))
        new_lambda = self.lambda_t * math.exp(clipped_step)
        self.lambda_t = float(np.clip(new_lambda, self.lambda_min, self.lambda_max))

        # ---------------------------------------------------------------------
        # 2. Section 3: Recursive Updates for Sufficient Statistics
        # ---------------------------------------------------------------------
        outer_xx = np.outer(x, x)

        if self.decay_all_arms and self.gamma < 1.0 - 1e-12:
            # Synchronous discount across all idle arms
            for a in range(self.n_arms):
                if a == arm_id:
                    continue
                self.A[a] = self.gamma * self.A[a] + (1.0 - self.gamma) * self.l2_lambda * eye
                self.b_reward[a] *= self.gamma
                self.b_cost[a] *= self.gamma

                if self.use_sherman_morrison:
                    # Invert directly or decay inverse towards (1 / lambda) * I
                    self.A_inv[a] = np.linalg.inv(self.A[a])
                else:
                    self.A_inv[a] = np.linalg.inv(self.A[a])

                self.theta[a] = np.dot(self.A_inv[a], self.b_reward[a])
                self.psi[a] = np.dot(self.A_inv[a], self.b_cost[a])

        # Update pulled arm
        if self.use_sherman_morrison and self.gamma >= 1.0 - 1e-12:
            # Standard stationary rank-1 Sherman-Morrison
            u = np.dot(self.A_inv[arm_id], x)
            denom = 1.0 + float(np.dot(x, u))
            self.A_inv[arm_id] -= np.outer(u, u) / max(1e-12, denom)
            self.A_inv[arm_id] = 0.5 * (self.A_inv[arm_id] + self.A_inv[arm_id].T)
            self.A[arm_id] += outer_xx
        elif self.use_sherman_morrison:
            # Discounted rank-1 Sherman-Morrison (Section 3 Formula)
            self.A_inv[arm_id] = self._sherman_morrison_rank1(self.A_inv[arm_id], x, self.gamma)
            self.A[arm_id] = self.gamma * self.A[arm_id] + outer_xx
        else:
            # Standard direct inversion
            self.A[arm_id] = self.gamma * self.A[arm_id] + outer_xx
            self.A_inv[arm_id] = np.linalg.inv(self.A[arm_id])

        # Update discounted response vectors
        self.b_reward[arm_id] = self.gamma * self.b_reward[arm_id] + r * x
        self.b_cost[arm_id] = self.gamma * self.b_cost[arm_id] + c * x

        # Update parameter estimators
        self.theta[arm_id] = np.dot(self.A_inv[arm_id], self.b_reward[arm_id])
        self.psi[arm_id] = np.dot(self.A_inv[arm_id], self.b_cost[arm_id])

        # ---------------------------------------------------------------------
        # 3. Telemetry & Counters
        # ---------------------------------------------------------------------
        self.total_rounds += 1
        self.arm_counts[arm_id] += 1
        self.cumulative_reward += r
        self.cumulative_cost += c

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
        """Explicit semantic convenience method for full CBwK feedback (x_t, a_t, r_t, c_t)."""
        self.update(arm_id=arm_id, context=context, reward=reward, cost=cost)

    # -------------------------------------------------------------------------
    # Stopping Condition (Section 4)
    # -------------------------------------------------------------------------
    def check_stopping_condition(
        self, context: Optional[np.ndarray] = None
    ) -> Tuple[bool, str]:
        """
        Section 4: Stopping Condition.
        Process terminates if:
          1. Total Iterations Completed: t >= T
          2. Budget Exhausted: B_t <= 0
          3. Insufficient Budget: B_t < min_{a in A} hat{c}_{t,a}(x_t)

        Returns
        -------
        Tuple[bool, str]
            (is_stopped, termination_reason)
        """
        if self.total_rounds >= self.total_horizon:
            return True, "horizon_completed"

        if self.remaining_budget <= 1e-6:
            return True, "budget_exhausted"

        if context is not None:
            x = self.validate_context(context)
            min_arm_cost = min(self.estimate_cost(a, x)[1] for a in range(self.n_arms))
            if self.remaining_budget < min_arm_cost:
                return True, "budget_below_min_arm_cost"

        return False, "active"

    # -------------------------------------------------------------------------
    # Diagnostics & Serialization
    # -------------------------------------------------------------------------
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic state for telemetries, logs, and comparative plots."""
        theta_norms = [float(np.linalg.norm(self.theta[a])) for a in range(self.n_arms)]
        psi_norms = [float(np.linalg.norm(self.psi[a])) for a in range(self.n_arms)]
        cer = (
            self.cumulative_reward / max(1e-5, self.cumulative_cost)
            if self.cumulative_cost > 0
            else 0.0
        )

        return {
            "policy_name": self.name,
            "total_rounds": self.total_rounds,
            "total_horizon": self.total_horizon,
            "initial_budget": self.initial_budget,
            "remaining_budget": round(self.remaining_budget, 6),
            "cumulative_cost": round(self.cumulative_cost, 6),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "mean_reward": (
                round(self.cumulative_reward / self.total_rounds, 4)
                if self.total_rounds > 0
                else 0.0
            ),
            "capital_efficiency_ratio": round(cer, 4),
            "current_lambda_dual": round(self.lambda_t, 6),
            "arm_counts": self.arm_counts.tolist(),
            "theta_l2_norms": theta_norms,
            "psi_l2_norms": psi_norms,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "beta": self.beta,
            "eta": round(self.eta, 6),
            "use_sherman_morrison": self.use_sherman_morrison,
        }

    def save_checkpoint(self, path: Path) -> None:
        """Persist policy state, dual parameters, and matrices to JSON."""
        data = {
            "name": self.name,
            "n_arms": self.n_arms,
            "context_dim": self.context_dim,
            "initial_budget": self.initial_budget,
            "remaining_budget": self.remaining_budget,
            "total_horizon": self.total_horizon,
            "total_rounds": self.total_rounds,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "beta": self.beta,
            "l2_lambda": self.l2_lambda,
            "lambda_t": self.lambda_t,
            "eta": self.eta,
            "cumulative_reward": self.cumulative_reward,
            "cumulative_cost": self.cumulative_cost,
            "arm_counts": self.arm_counts.tolist(),
            "A": self.A.tolist(),
            "A_inv": self.A_inv.tolist(),
            "b_reward": self.b_reward.tolist(),
            "theta": self.theta.tolist(),
            "b_cost": self.b_cost.tolist(),
            "psi": self.psi.tolist(),
            "dual_history": self.dual_history,
            "budget_history": self.budget_history,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def load_checkpoint(self, path: Path) -> None:
        """Load policy state from saved JSON checkpoint."""
        data = json.loads(path.read_text())
        self.n_arms = int(data["n_arms"])
        self.context_dim = int(data["context_dim"])
        self.initial_budget = float(data["initial_budget"])
        self.remaining_budget = float(data["remaining_budget"])
        self.total_horizon = int(data["total_horizon"])
        self.total_rounds = int(data["total_rounds"])
        self.gamma = float(data["gamma"])
        self.alpha = float(data["alpha"])
        self.beta = float(data["beta"])
        self.l2_lambda = float(data["l2_lambda"])
        self.lambda_t = float(data["lambda_t"])
        self.eta = float(data["eta"])
        self.cumulative_reward = float(data["cumulative_reward"])
        self.cumulative_cost = float(data["cumulative_cost"])
        self.arm_counts = np.array(data["arm_counts"], dtype=np.int64)
        self.A = np.array(data["A"], dtype=np.float64)
        self.A_inv = np.array(data["A_inv"], dtype=np.float64)
        self.b_reward = np.array(data["b_reward"], dtype=np.float64)
        self.theta = np.array(data["theta"], dtype=np.float64)
        self.b_cost = np.array(data["b_cost"], dtype=np.float64)
        self.psi = np.array(data["psi"], dtype=np.float64)
        self.dual_history = data.get("dual_history", [self.lambda_t])
        self.budget_history = data.get("budget_history", [self.remaining_budget])


# Alias for intuitive readability
DiscountedLinUCBWithKnapsack = DiscountedLinCBwK

