"""
Unit test suite for Dynamic Regret Evaluator (D-LinCBwK).
"""

import math
import tempfile
from pathlib import Path
import numpy as np
import pytest

from cmab_anvil.algorithms.dynamic_regret import (
    DynamicRegretEvaluator,
    DynamicRegretStep,
    DynamicRegretSummary,
    compare_policies_dynamic_regret,
)


def test_evaluator_initialization():
    ev = DynamicRegretEvaluator(
        context_dim=12,
        n_arms=6,
        initial_budget=0.50,
        policy_name="D-LinCBwK",
    )
    assert ev.context_dim == 12
    assert ev.n_arms == 6
    assert ev.initial_budget == 0.50
    assert ev.state_switches == 1
    assert ev.total_rounds == 0
    assert ev.cumulative_regret == 0.0


def test_online_step_regret_tracking():
    ev = DynamicRegretEvaluator(
        context_dim=10,
        n_arms=4,
        initial_budget=1.0,
        oracle_reward_ceiling=0.90,
    )

    # Step 1: Agent receives 0.60, optimal is 0.90 -> delta_1 = 0.30
    step1 = ev.step(round_idx=1, chosen_arm=0, reward=0.60, sstore_executed=False, gas_cost=0.05)
    assert math.isclose(step1.instantaneous_regret, 0.30, rel_tol=1e-5)
    assert math.isclose(step1.cumulative_regret, 0.30, rel_tol=1e-5)
    assert step1.state_switches == 1
    assert math.isclose(step1.remaining_budget, 0.95, rel_tol=1e-5)

    # Step 2: SSTORE executed (persistent state change), counterfactual arms provided
    # All arms rewards = {0: 0.20, 1: 0.85, 2: 0.10, 3: 0.40} -> optimal = 0.85
    # Agent chose arm 2 (0.10) -> delta_2 = 0.75
    step2 = ev.step(
        round_idx=2,
        chosen_arm=2,
        reward=0.10,
        all_arm_rewards={0: 0.20, 1: 0.85, 2: 0.10, 3: 0.40},
        sstore_executed=True,
        gas_cost=0.10,
    )
    assert step2.state_switches == 2
    assert math.isclose(step2.optimal_reward, 0.85, rel_tol=1e-5)
    assert math.isclose(step2.instantaneous_regret, 0.75, rel_tol=1e-5)
    assert math.isclose(step2.cumulative_regret, 0.30 + 0.75, rel_tol=1e-5)
    assert math.isclose(step2.remaining_budget, 0.85, rel_tol=1e-5)


def test_theoretical_bound_formulas():
    d = 10
    k = 5
    b = 0.50
    t = 24
    s = 4
    opt = 18.0

    ev = DynamicRegretEvaluator(context_dim=d, n_arms=k, initial_budget=b)

    # 1. Unconstrained: O(d * sqrt(S * T * log(T)))
    unconstrained = ev.compute_unconstrained_bound(t=t, s=s, d=d)
    expected_uncon = d * math.sqrt(s * t * math.log(t + 1.0))
    assert math.isclose(unconstrained, expected_uncon, rel_tol=1e-5)

    # 2. Knapsack: O((OPT / B) * d * sqrt(S * T * log(d * T * K)))
    knapsack = ev.compute_knapsack_bound(t=t, s=s, opt=opt, budget=b, d=d, k=k)
    expected_knap = (opt / b) * d * math.sqrt(s * t * math.log(d * t * k + 1.0))
    assert math.isclose(knapsack, expected_knap, rel_tol=1e-5)


def test_batch_trajectory_evaluation():
    ev = DynamicRegretEvaluator(context_dim=8, n_arms=3, initial_budget=1.0)

    round_logs = [
        {"round": 1, "arm_id": 0, "reward": 0.5, "gas_cost": 0.02, "sstore_executed": False},
        {"round": 2, "arm_id": 1, "reward": 0.8, "gas_cost": 0.03, "sstore_executed": True},
        {"round": 3, "arm_id": 1, "reward": 0.9, "gas_cost": 0.02, "sstore_executed": False},
        {"round": 4, "arm_id": 2, "reward": 0.7, "gas_cost": 0.04, "sstore_executed": True},
    ]

    summary: DynamicRegretSummary = ev.evaluate_trajectory(round_logs)

    assert summary.total_rounds == 4
    assert summary.state_switches == 3  # 1 initial + 2 SSTOREs
    assert math.isclose(summary.cumulative_reward, 0.5 + 0.8 + 0.9 + 0.7)
    assert summary.cumulative_optimal_opt >= summary.cumulative_reward
    assert summary.cumulative_dynamic_regret >= 0.0

    # Test dictionary export
    d = summary.to_dict()
    assert d["total_rounds"] == 4
    assert d["state_switches_S"] == 3
    assert "theoretical_bound_knapsack" in d


def test_plot_generation():
    ev = DynamicRegretEvaluator(context_dim=6, n_arms=3, initial_budget=0.5)
    for t in range(1, 10):
        ev.step(
            round_idx=t,
            chosen_arm=t % 3,
            reward=0.3 + 0.05 * t,
            sstore_executed=(t in (3, 7)),
            gas_cost=0.015,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = Path(tmpdir) / "dynamic_regret_plot.png"
        fig = ev.plot_regret(save_path=plot_path)
        assert plot_path.exists()
        assert plot_path.stat().st_size > 1000


def test_comparative_policy_evaluation():
    logs_linucb = [
        {"round": t, "arm_id": 0, "reward": 0.30, "gas_cost": 0.02, "sstore_executed": (t % 4 == 0)}
        for t in range(1, 9)
    ]
    logs_d_lincbwk = [
        {"round": t, "arm_id": 1, "reward": 0.75, "gas_cost": 0.01, "sstore_executed": (t % 4 == 0)}
        for t in range(1, 9)
    ]

    policies = {
        "LinUCB": logs_linucb,
        "D-LinCBwK": logs_d_lincbwk,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cmp_plot = Path(tmpdir) / "comparative_regret.png"
        summaries = compare_policies_dynamic_regret(
            policies_round_logs=policies,
            context_dim=8,
            n_arms=3,
            initial_budget=0.5,
            plot_save_path=cmp_plot,
        )

        assert "LinUCB" in summaries
        assert "D-LinCBwK" in summaries
        # D-LinCBwK collected higher rewards -> should have lower dynamic regret
        assert summaries["D-LinCBwK"].cumulative_dynamic_regret < summaries["LinUCB"].cumulative_dynamic_regret
        assert cmp_plot.exists()

