"""
Dynamic Regret (Tracking Regret) Evaluator for Non-Stationary EVM Bandits (D-LinCBwK).

Theoretical Foundation:
    In stateful smart contract fuzzing, tracking a single static best arm is futile
    because persistent EVM Storage evolves continuously (via SSTORE). An optimal agent
    must follow a dynamic sequence of optimal actions (a_1^*, a_2^*, ..., a_T^*) matching
    the contract's internal state transitions [smartianFuzz.pdf, AuditorKnwlLrnFuzz.pdf].

1. Dynamic Regret (Tracking Regret) Formulation:
    R_T^{dynamic} = E[ sum_{t=1}^T r_t(a_t^*) ] - E[ sum_{t=1}^T r_t(a_t) ]
                  = OPT - sum_{t=1}^T r_t(a_t)

    where:
      - a_t^* = argmax_{a in A} r_t(a) is the optimal mutator at round t matching current EVM state.
      - OPT = sum_{t=1}^T r_t(a_t^*) is the cumulative reward of the optimal policy in hindsight.
      - delta_t = r_t(a_t^*) - r_t(a_t) is the instantaneous regret at round t.

2. Influence of Deterministic EVM Execution (eta_t = 0):
    Unlike stochastic bandits with Gaussian noise eta_t ~ N(0, sigma^2), EVM state transitions
    and execution traces are completely deterministic (eta_t = 0) given identical inputs and state.
    - Zero Random Sample Variance: No need to pull identical arms repeatedly just to average out noise.
    - Rapid Confidence Shrinkage: The confidence radius shrinks much faster, bounded purely
      by linear estimation uncertainty across the d-dimensional context space.

3. Theoretical Dynamic Regret Bounds:
    - Standard Unconstrained Dynamic Regret:
        R_T^{dynamic} = O( d * sqrt( S * T * log(T) ) )

    - Contextual Bandits with Knapsacks (CBwK) with Global Gas/Capital Constraint (B):
        R_T^{dynamic} = O( (OPT / B) * d * sqrt( S * T * log(d * T * |A|) ) )

4. Component Parameters:
    - d   : Dimension of pure EVM context vector (e.g., d = 10, 12, 78).
    - S   : Number of persistent state switches / storage modifications (tracked via SSTORE).
    - T   : Total interaction rounds / fuzzing transactions.
    - B   : Total global gas/capital budget.
    - OPT : Ideal cumulative reward collected by optimal sequence in hindsight.
    - |A| : Number of discrete arms / mutators (K).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import math
from pathlib import Path
import numpy as np


@dataclass
class DynamicRegretStep:
    """Individual round snapshot of dynamic regret tracking."""
    round_idx: int
    chosen_arm: int
    reward: float
    optimal_reward: float
    instantaneous_regret: float
    cumulative_regret: float
    cumulative_reward: float
    cumulative_optimal: float
    sstore_executed: bool
    state_switches: int
    gas_cost: float
    remaining_budget: float
    theoretical_bound_unconstrained: float
    theoretical_bound_knapsack: float
    sublinear_ratio_knapsack: float


@dataclass
class DynamicRegretSummary:
    """Consolidated evaluation summary for a policy over a full trajectory."""
    policy_name: str
    total_rounds: int
    context_dim: int
    n_arms: int
    initial_budget: float
    remaining_budget: float
    state_switches: int
    cumulative_reward: float
    cumulative_optimal_opt: float
    cumulative_dynamic_regret: float
    mean_dynamic_regret: float
    theoretical_bound_unconstrained: float
    theoretical_bound_knapsack: float
    sublinear_ratio_unconstrained: float
    sublinear_ratio_knapsack: float
    is_sublinear: bool
    average_gas_cost: float
    steps: List[DynamicRegretStep] = field(default_factory=list)

    @property
    def trajectory(self) -> List[DynamicRegretStep]:
        return self.steps

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary into serializable dictionary."""
        return {
            "policy_name": self.policy_name,
            "total_rounds": self.total_rounds,
            "context_dim": self.context_dim,
            "n_arms": self.n_arms,
            "initial_budget": self.initial_budget,
            "remaining_budget": round(self.remaining_budget, 6),
            "state_switches_S": self.state_switches,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "cumulative_optimal_OPT": round(self.cumulative_optimal_opt, 4),
            "cumulative_dynamic_regret": round(self.cumulative_dynamic_regret, 4),
            "mean_dynamic_regret": round(self.mean_dynamic_regret, 6),
            "theoretical_bound_unconstrained": round(self.theoretical_bound_unconstrained, 4),
            "theoretical_bound_knapsack": round(self.theoretical_bound_knapsack, 4),
            "sublinear_ratio_unconstrained": round(self.sublinear_ratio_unconstrained, 4),
            "sublinear_ratio_knapsack": round(self.sublinear_ratio_knapsack, 4),
            "is_sublinear": self.is_sublinear,
            "average_gas_cost": round(self.average_gas_cost, 6),
        }


class DynamicRegretEvaluator:
    """
    Evaluator for Dynamic Regret (Tracking Regret) in EVM Smart Contract Fuzzing.

    Features:
      - Real-time online round-by-round tracking.
      - Trajectory batch evaluation from round logs or result JSONs.
      - Theoretical bound calculation (unconstrained vs. Knapsack CBwK).
      - Sublinear convergence checking.
      - Publication-grade multi-panel comparative visualization.
    """

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        initial_budget: float = 1.0,
        policy_name: str = "D-LinCBwK",
        oracle_reward_ceiling: float = 1.0,
        bound_constant_c: float = 1.0,
    ):
        """
        Parameters
        ----------
        context_dim : int
            Dimension d of EVM context vector x_t.
        n_arms : int
            Number of arms K = |A|.
        initial_budget : float
            Total initial gas / capital budget B.
        policy_name : str
            Name of policy under evaluation.
        oracle_reward_ceiling : float
            Default optimal reward ceiling when counterfactual rewards are absent (default: 1.0).
        bound_constant_c : float
            Leading constant C in O(...) asymptotic bound formulation.
        """
        if context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        if initial_budget <= 0.0:
            raise ValueError("initial_budget must be positive.")

        self.context_dim = int(context_dim)
        self.n_arms = int(n_arms)
        self.initial_budget = float(initial_budget)
        self.policy_name = str(policy_name)
        self.oracle_ceiling = float(oracle_reward_ceiling)
        self.bound_c = float(bound_constant_c)

        # State accumulators
        self.total_rounds: int = 0
        self.state_switches: int = 1  # Base epoch S starts at 1
        self.cumulative_reward: float = 0.0
        self.cumulative_optimal: float = 0.0
        self.cumulative_regret: float = 0.0
        self.cumulative_gas_cost: float = 0.0
        self.remaining_budget: float = self.initial_budget

        self.steps: List[DynamicRegretStep] = []

    def reset(self) -> None:
        """Reset all trajectory histories and accumulators."""
        self.total_rounds = 0
        self.state_switches = 1
        self.cumulative_reward = 0.0
        self.cumulative_optimal = 0.0
        self.cumulative_regret = 0.0
        self.cumulative_gas_cost = 0.0
        self.remaining_budget = self.initial_budget
        self.steps.clear()

    # -------------------------------------------------------------------------
    # Theoretical Bound Calculations (Section 3)
    # -------------------------------------------------------------------------
    def compute_unconstrained_bound(
        self,
        t: int,
        s: int,
        d: Optional[int] = None,
    ) -> float:
        """
        Compute standard unconstrained non-stationary Dynamic Regret Bound:
            R_T^{dynamic} = O( d * sqrt( S * T * log(T) ) )
        """
        dim = float(d or self.context_dim)
        rounds = max(1, t)
        switches = max(1, s)
        # Note: log(rounds + 1) ensures well-defined positive argument when t=1
        log_term = math.log(rounds + 1.0)
        return float(self.bound_c * dim * math.sqrt(switches * rounds * log_term))

    def compute_knapsack_bound(
        self,
        t: int,
        s: int,
        opt: float,
        budget: Optional[float] = None,
        d: Optional[int] = None,
        k: Optional[int] = None,
    ) -> float:
        """
        Compute Knapsack Dynamic Regret Bound with Global Gas Budget constraint B:
            R_T^{dynamic} = O( (OPT / B) * d * sqrt( S * T * log(d * T * |A|) ) )
        """
        dim = float(d or self.context_dim)
        arms = float(k or self.n_arms)
        b_val = float(budget or self.initial_budget)
        rounds = max(1, t)
        switches = max(1, s)
        opt_val = max(1e-4, float(opt))

        budget_ratio = opt_val / max(1e-4, b_val)
        log_term = math.log(dim * rounds * arms + 1.0)
        core_growth = dim * math.sqrt(switches * rounds * log_term)

        return float(self.bound_c * budget_ratio * core_growth)

    # -------------------------------------------------------------------------
    # Online Stepping
    # -------------------------------------------------------------------------
    def step(
        self,
        round_idx: int,
        chosen_arm: int,
        reward: float,
        optimal_reward: Optional[float] = None,
        all_arm_rewards: Optional[Union[Dict[int, float], List[float]]] = None,
        sstore_executed: bool = False,
        gas_cost: float = 0.0,
    ) -> DynamicRegretStep:
        """
        Record a single fuzzing transaction round and compute instantaneous & cumulative regret.

        Parameters
        ----------
        round_idx : int
            Round number t >= 1.
        chosen_arm : int
            Index of arm executed by policy.
        reward : float
            Bandit reward r_t(a_t) received.
        optimal_reward : Optional[float]
            Counterfactual optimal reward r_t(a_t^*) for current state.
            If None, inferred from `all_arm_rewards` or defaults to max(reward, oracle_ceiling).
        all_arm_rewards : Optional[Union[Dict[int, float], List[float]]]
            Full vector or dictionary of rewards across all arms under current state.
        sstore_executed : bool
            True if transaction executed SSTORE (persistent EVM state mutation).
        gas_cost : float
            Gas cost c_t consumed by transaction.
        """
        t = int(round_idx)
        r_actual = float(np.clip(reward, 0.0, 1.0))
        c_cost = max(0.0, float(gas_cost))

        # 1. Determine optimal reward r_t(a_t^*)
        if optimal_reward is not None:
            r_opt = float(np.clip(optimal_reward, r_actual, 1.0))
        elif all_arm_rewards is not None:
            values = (
                list(all_arm_rewards.values())
                if isinstance(all_arm_rewards, dict)
                else list(all_arm_rewards)
            )
            r_opt = float(max(values)) if values else r_actual
        else:
            # Oracle assumption in smart contract testing: an optimal state-aligned mutator
            # extracts the ceiling utility (1.0), or at least the realized reward.
            r_opt = max(r_actual, self.oracle_ceiling)

        # 2. Instantaneous regret: delta_t = r_t(a_t^*) - r_t(a_t)
        instant_regret = max(0.0, r_opt - r_actual)

        # 3. Track state switches S (SSTORE)
        if sstore_executed:
            self.state_switches += 1

        # 4. Update accumulators
        self.total_rounds += 1
        self.cumulative_reward += r_actual
        self.cumulative_optimal += r_opt
        self.cumulative_regret += instant_regret
        self.cumulative_gas_cost += c_cost
        self.remaining_budget = max(0.0, self.remaining_budget - c_cost)

        # 5. Theoretical bounds at round t
        bound_unconstrained = self.compute_unconstrained_bound(
            t=self.total_rounds,
            s=self.state_switches,
        )
        bound_knapsack = self.compute_knapsack_bound(
            t=self.total_rounds,
            s=self.state_switches,
            opt=self.cumulative_optimal,
        )
        sublin_ratio = (
            self.cumulative_regret / bound_knapsack
            if bound_knapsack > 1e-6
            else 0.0
        )

        step_record = DynamicRegretStep(
            round_idx=t,
            chosen_arm=chosen_arm,
            reward=r_actual,
            optimal_reward=r_opt,
            instantaneous_regret=instant_regret,
            cumulative_regret=self.cumulative_regret,
            cumulative_reward=self.cumulative_reward,
            cumulative_optimal=self.cumulative_optimal,
            sstore_executed=sstore_executed,
            state_switches=self.state_switches,
            gas_cost=c_cost,
            remaining_budget=self.remaining_budget,
            theoretical_bound_unconstrained=bound_unconstrained,
            theoretical_bound_knapsack=bound_knapsack,
            sublinear_ratio_knapsack=sublin_ratio,
        )
        self.steps.append(step_record)
        return step_record

    # -------------------------------------------------------------------------
    # Batch Trajectory Evaluation
    # -------------------------------------------------------------------------
    def evaluate_trajectory(
        self,
        round_logs: List[Dict[str, Any]],
        optimal_rewards: Optional[List[float]] = None,
        horizon_t: Optional[int] = None,
    ) -> DynamicRegretSummary:
        """
        Evaluate full interaction trajectory from round logs (Persamaan 3.22).

        If the policy experienced premature budget exhaustion at stopping time tau < T
        (where T = `horizon_t`), then for all remaining rounds t in {tau + 1, ..., T},
        the policy receives r_t(a_t) = 0.0 and incurs full regret r_t^* (Halaman 47-48).

        Parameters
        ----------
        round_logs : List[Dict[str, Any]]
            Sequential round log entries (e.g. from `run_policy_on_contract`).
        optimal_rewards : Optional[List[float]]
            Optional counterfactual optimal rewards per round.
        horizon_t : Optional[int]
            Planned total horizon T. If provided and > len(round_logs), pads post-bankruptcy
            rounds tau+1..T with r_t(a_t) = 0.0.
        """
        self.reset()
        for idx, entry in enumerate(round_logs):
            t = entry.get("round", idx + 1)
            arm_id = entry.get("arm_id", 0)
            rew = float(entry.get("reward", 0.0))
            cost = float(entry.get("cost_eth", entry.get("burned_eth", entry.get("gas_cost", 0.0))))
            sstore = bool(entry.get("sstore_executed", False))

            r_opt = optimal_rewards[idx] if (optimal_rewards and idx < len(optimal_rewards)) else None
            self.step(
                round_idx=t,
                chosen_arm=arm_id,
                reward=rew,
                optimal_reward=r_opt,
                sstore_executed=sstore,
                gas_cost=cost,
            )

        # Handle premature stopping tau < T (Persamaan 3.22 & Sub-bab 3.6 Butir 1):
        # For t in {tau + 1, ..., T}, r_t(a_t) = 0 and instantaneous regret = r_t^*
        tau = len(round_logs)
        if horizon_t is not None and horizon_t > tau:
            for idx in range(tau, horizon_t):
                t = idx + 1
                r_opt = optimal_rewards[idx] if (optimal_rewards and idx < len(optimal_rewards)) else None
                self.step(
                    round_idx=t,
                    chosen_arm=-1,
                    reward=0.0,
                    optimal_reward=r_opt,
                    sstore_executed=False,
                    gas_cost=0.0,
                )

        return self.get_summary()

    def get_summary(self) -> DynamicRegretSummary:
        """Compile consolidated Dynamic Regret summary."""
        T = max(1, self.total_rounds)
        mean_regret = self.cumulative_regret / float(T)

        bound_unconstrained = self.compute_unconstrained_bound(T, self.state_switches)
        bound_knapsack = self.compute_knapsack_bound(T, self.state_switches, self.cumulative_optimal)

        ratio_unconstrained = (
            self.cumulative_regret / bound_unconstrained
            if bound_unconstrained > 1e-6
            else 0.0
        )
        ratio_knapsack = (
            self.cumulative_regret / bound_knapsack
            if bound_knapsack > 1e-6
            else 0.0
        )

        # Sublinear verification: Mean regret R_T / T should decline or remain bounded
        is_sublin = True
        if len(self.steps) >= 8:
            first_half_mean = float(np.mean([s.instantaneous_regret for s in self.steps[: len(self.steps) // 2]]))
            second_half_mean = float(np.mean([s.instantaneous_regret for s in self.steps[len(self.steps) // 2 :]]))
            is_sublin = second_half_mean <= first_half_mean + 0.05

        return DynamicRegretSummary(
            policy_name=self.policy_name,
            total_rounds=self.total_rounds,
            context_dim=self.context_dim,
            n_arms=self.n_arms,
            initial_budget=self.initial_budget,
            remaining_budget=self.remaining_budget,
            state_switches=self.state_switches,
            cumulative_reward=self.cumulative_reward,
            cumulative_optimal_opt=self.cumulative_optimal,
            cumulative_dynamic_regret=self.cumulative_regret,
            mean_dynamic_regret=mean_regret,
            theoretical_bound_unconstrained=bound_unconstrained,
            theoretical_bound_knapsack=bound_knapsack,
            sublinear_ratio_unconstrained=ratio_unconstrained,
            sublinear_ratio_knapsack=ratio_knapsack,
            is_sublinear=is_sublin,
            average_gas_cost=self.cumulative_gas_cost / float(T),
            steps=list(self.steps),
        )

    # -------------------------------------------------------------------------
    # Visualization & Plotting
    # -------------------------------------------------------------------------
    def plot_regret(
        self,
        save_path: Optional[Union[str, Path]] = None,
        title_prefix: str = "EVM Fuzzing Dynamic Regret Evaluation",
    ) -> Any:
        """
        Generate publication-grade 4-panel diagnostic chart of Dynamic Regret:
          - Panel 1: Cumulative Dynamic Regret vs. Theoretical CBwK Bound.
          - Panel 2: Instantaneous Regret & EVM State Switches (SSTORE).
          - Panel 3: Average Regret Convergence (R_t / t).
          - Panel 4: Cumulative Policy Reward vs. Ideal Hindsight OPT.
        """
        import matplotlib.pyplot as plt

        summary = self.get_summary()
        rounds = [s.round_idx for s in self.steps]
        cum_regrets = [s.cumulative_regret for s in self.steps]
        inst_regrets = [s.instantaneous_regret for s in self.steps]
        knapsack_bounds = [s.theoretical_bound_knapsack for s in self.steps]
        unconstrained_bounds = [s.theoretical_bound_unconstrained for s in self.steps]
        cum_rewards = [s.cumulative_reward for s in self.steps]
        cum_opts = [s.cumulative_optimal for s in self.steps]
        mean_regrets = [cum_regrets[i] / float(max(1, rounds[i])) for i in range(len(rounds))]
        sstore_flags = [1 if s.sstore_executed else 0 for s in self.steps]

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"{title_prefix} | {self.policy_name}\n"
            f"(d={self.context_dim}, K={self.n_arms}, B={self.initial_budget:.2f} ETH, S={self.state_switches})",
            fontsize=14,
            fontweight="bold",
        )

        # Panel 1: Cumulative Dynamic Regret vs Theoretical Bound
        ax1 = axs[0, 0]
        ax1.plot(rounds, cum_regrets, label=f"Empirical Dynamic Regret $R_t^{{dynamic}}$", color="#1f77b4", lw=2.5)
        ax1.plot(rounds, knapsack_bounds, label=r"CBwK Bound $\mathcal{O}\left(\frac{OPT}{B} d \sqrt{S T \log(dTK)}\right)$", color="#d62728", ls="--", lw=2.0)
        ax1.plot(rounds, unconstrained_bounds, label=r"Unconstrained Bound $\mathcal{O}\left(d \sqrt{S T \log T}\right)$", color="#2ca02c", ls=":", lw=1.8)
        ax1.set_title("1. Cumulative Dynamic Regret vs Theoretical Bounds", fontweight="bold")
        ax1.set_xlabel("Rounds $t$")
        ax1.set_ylabel("Cumulative Regret")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")

        # Panel 2: Instantaneous Regret & EVM State Switches (SSTORE)
        ax2 = axs[0, 1]
        ax2.bar(rounds, inst_regrets, color="#ff7f0e", alpha=0.6, label=r"Instantaneous Regret $\delta_t = r_t^* - r_t$")
        switch_rounds = [rounds[i] for i, flag in enumerate(sstore_flags) if flag == 1]
        if switch_rounds:
            ax2.scatter(
                switch_rounds,
                [max(inst_regrets) * 1.05] * len(switch_rounds),
                color="#9467bd",
                marker="v",
                s=70,
                label=f"EVM State Switch (SSTORE, S={len(switch_rounds)})",
                zorder=5,
            )
        ax2.set_title("2. Instantaneous Regret & State Transition Switches", fontweight="bold")
        ax2.set_xlabel("Rounds $t$")
        ax2.set_ylabel("Regret per Round")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")

        # Panel 3: Average Regret Convergence (No-Regret Check: R_t / t -> 0)
        ax3 = axs[1, 0]
        ax3.plot(rounds, mean_regrets, color="#8c564b", lw=2.2, label=r"Average Regret $\bar{R}_t = R_t / t$")
        ax3.axhline(0.0, color="gray", ls="--", alpha=0.5)
        ax3.set_title(r"3. Sublinear Convergence Rate ($\lim_{t \to \infty} \bar{R}_t \to 0$)", fontweight="bold")
        ax3.set_xlabel("Rounds $t$")
        ax3.set_ylabel(r"Average Regret $\bar{R}_t$")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right")

        # Panel 4: Policy Cumulative Reward vs Optimal Target (OPT)
        ax4 = axs[1, 1]
        ax4.plot(rounds, cum_opts, color="#2ca02c", lw=2.2, label=r"Ideal Hindsight Optimal $OPT_t$")
        ax4.plot(rounds, cum_rewards, color="#1f77b4", lw=2.2, label=f"Policy Reward ({self.policy_name})")
        ax4.fill_between(rounds, cum_rewards, cum_opts, color="#d62728", alpha=0.15, label="Regret Gap Area")
        ax4.set_title("4. Cumulative Reward vs Hindsight Target OPT", fontweight="bold")
        ax4.set_xlabel("Rounds $t$")
        ax4.set_ylabel("Cumulative Reward")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="upper left")

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=300)
            print(f"[*] Dynamic Regret chart saved to: {p}")
        return fig


# -----------------------------------------------------------------------------
# Multi-Policy Comparative Regret Evaluator
# -----------------------------------------------------------------------------
def compare_policies_dynamic_regret(
    policies_round_logs: Dict[str, List[Dict[str, Any]]],
    context_dim: int,
    n_arms: int,
    initial_budget: float = 1.0,
    optimal_rewards: Optional[List[float]] = None,
    oracle_reward_ceiling: float = 1.00,
    horizon_t: Optional[int] = None,
    plot_save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, DynamicRegretSummary]:
    """
    Perform comparative Dynamic Regret evaluation across multiple CMAB policies
    (e.g., UniformRandom, LinUCB, D-LinUCB, LinCBwK, D-LinCBwK).
    """
    import matplotlib.pyplot as plt

    max_len = max((len(logs) for logs in policies_round_logs.values()), default=0)
    effective_horizon = horizon_t if (horizon_t is not None and horizon_t >= max_len) else max_len

    # If optimal_rewards is not provided, compute CRN cross-policy envelope r_t^* (Sub-bab 3.6 Butir 1)
    if optimal_rewards is None and effective_horizon > 0:
        crn_envelope: List[float] = []
        for idx in range(effective_horizon):
            round_vals = [
                float(logs[idx].get("reward", 0.0))
                for logs in policies_round_logs.values()
                if idx < len(logs)
            ]
            # Under CRN, the instantaneous oracle achieves at least the best observed reward across
            # policies at round t (or prerequisite state shaping ceiling 0.50 when all policies reverted)
            best_obs = max(round_vals) if round_vals else 0.50
            crn_envelope.append(max(0.50, best_obs))
        optimal_rewards = crn_envelope

    summaries: Dict[str, DynamicRegretSummary] = {}
    evaluators: Dict[str, DynamicRegretEvaluator] = {}

    for name, logs in policies_round_logs.items():
        ev = DynamicRegretEvaluator(
            context_dim=context_dim,
            n_arms=n_arms,
            initial_budget=initial_budget,
            oracle_reward_ceiling=oracle_reward_ceiling,
            policy_name=name,
        )
        summary = ev.evaluate_trajectory(
            logs,
            optimal_rewards=optimal_rewards,
            horizon_t=effective_horizon,
        )
        summaries[name] = summary
        evaluators[name] = ev

    if plot_save_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle(
            f"Comparative Dynamic Regret Evaluation (d={context_dim}, K={n_arms}, B={initial_budget:.2f} ETH)",
            fontsize=13,
            fontweight="bold",
        )

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for idx, (name, ev) in enumerate(evaluators.items()):
            c = colors[idx % len(colors)]
            rounds = [s.round_idx for s in ev.steps]
            cum_regrets = [s.cumulative_regret for s in ev.steps]
            mean_regrets = [cum_regrets[i] / float(max(1, rounds[i])) for i in range(len(rounds))]

            ax1.plot(rounds, cum_regrets, label=fr"{name} ($R_T={cum_regrets[-1]:.2f}$)", lw=2.2, color=c)
            ax2.plot(rounds, mean_regrets, label=fr"{name} ($\bar{{R}}={mean_regrets[-1]:.3f}$)", lw=2.0, color=c)

        # Plot theoretical Knapsack bound reference from first evaluator
        first_ev = list(evaluators.values())[0]
        rounds = [s.round_idx for s in first_ev.steps]
        knapsack_bounds = [s.theoretical_bound_knapsack for s in first_ev.steps]
        ax1.plot(rounds, knapsack_bounds, label="Theoretical CBwK Bound", ls="--", color="black", lw=1.5, alpha=0.7)

        ax1.set_title("Cumulative Dynamic Regret $R_t^{dynamic}$", fontweight="bold")
        ax1.set_xlabel("Rounds $t$")
        ax1.set_ylabel("Cumulative Regret")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_title(r"Average Regret $\bar{R}_t = R_t / t$", fontweight="bold")
        ax2.set_xlabel("Rounds $t$")
        ax2.set_ylabel(r"$\bar{R}_t$")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        p = Path(plot_save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
        print(f"[*] Comparative Dynamic Regret chart saved to: {p}")

    return summaries
