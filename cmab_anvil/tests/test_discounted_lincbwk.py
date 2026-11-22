"""
Unit test suite for Discounted Contextual Multi-Armed Bandits with Knapsacks (D-LinCBwK).
"""

import math
import tempfile
from pathlib import Path
import numpy as np
import pytest

from cmab_anvil.algorithms.discounted_linucb_knapsack import (
    DiscountedLinCBwK,
    DiscountedLinUCBWithKnapsack,
)
from cmab_anvil.algorithms.base import ArmSelectionResult


def test_initialization():
    policy = DiscountedLinCBwK(
        n_arms=4,
        context_dim=10,
        initial_budget=0.50,
        total_horizon=20,
        gamma=0.85,
        alpha=0.20,
        l2_lambda=1.0,
        initial_lambda=1.0,
    )
    assert policy.n_arms == 4
    assert policy.context_dim == 10
    assert policy.remaining_budget == 0.50
    assert policy.lambda_t == 1.0
    assert policy.total_rounds == 0
    assert math.isclose(policy.eta, math.sqrt(math.log(4) / 20), rel_tol=1e-5)
    assert np.allclose(policy.A[0], np.eye(10))
    assert np.allclose(policy.A_inv[0], np.eye(10))


def test_reward_and_cost_estimation():
    K, d = 3, 5
    policy = DiscountedLinCBwK(n_arms=K, context_dim=d, alpha=0.5, l2_lambda=1.0)
    x = np.array([1.0, 0.2, 0.5, 0.0, 0.1], dtype=np.float64)

    # Initial estimates with prior (theta=0, psi=0, A_inv=I)
    mean_r, unc_r, ucb_r = policy.estimate_reward(0, x)
    assert mean_r == 0.0
    expected_unc = 0.5 * math.sqrt(float(np.dot(x, x)))
    assert math.isclose(unc_r, expected_unc, rel_tol=1e-5)
    assert math.isclose(ucb_r, expected_unc, rel_tol=1e-5)

    # Cost estimate bounded by min_cost (0.001)
    exp_c, pred_c = policy.estimate_cost(0, x)
    assert exp_c == 0.0
    assert pred_c == policy.min_cost


def test_sherman_morrison_exactness():
    """
    Verify that the recursive Sherman-Morrison rank-1 update formula
    matches exact numpy matrix inversion within 1e-7 precision.
    """
    d = 6
    gamma = 0.82
    policy = DiscountedLinCBwK(
        n_arms=2,
        context_dim=d,
        gamma=gamma,
        l2_lambda=2.0,
        use_sherman_morrison=True,
    )

    rng = np.random.default_rng(123)
    x = rng.standard_normal(d)
    x /= np.linalg.norm(x)

    A_old = policy.A[0].copy()
    A_inv_old = policy.A_inv[0].copy()

    # Apply Sherman-Morrison
    A_inv_sm = policy._sherman_morrison_rank1(A_inv_old, x, gamma)

    # Compute ground truth matrix inversion: (gamma * A_old + x x^T)^(-1)
    A_new_exact = gamma * A_old + np.outer(x, x)
    A_inv_exact = np.linalg.inv(A_new_exact)

    np.testing.assert_allclose(A_inv_sm, A_inv_exact, rtol=1e-6, atol=1e-6)


def test_dual_multiplier_exponential_update():
    """
    Verify Stage C:
      If cost > B/T: lambda_t increases (penalizing gas).
      If cost < B/T: lambda_t decreases (encouraging action).
    """
    B, T = 1.0, 10
    target_rate = B / T  # 0.10
    policy = DiscountedLinCBwK(
        n_arms=2,
        context_dim=4,
        initial_budget=B,
        total_horizon=T,
        initial_lambda=1.0,
        dual_update_mode="observed",
    )

    x = np.ones(4) / 2.0

    # Case 1: Expensive transaction (cost = 0.25 > 0.10)
    policy.update(arm_id=0, context=x, reward=0.8, cost=0.25)
    assert policy.lambda_t > 1.0, "Lambda should increase when cost > target rate"
    assert math.isclose(policy.remaining_budget, 1.0 - 0.25)

    lambda_after_exp = policy.lambda_t

    # Case 2: Cheap transaction (cost = 0.01 < 0.10)
    policy.update(arm_id=0, context=x, reward=0.5, cost=0.01)
    assert policy.lambda_t < lambda_after_exp, "Lambda should decrease when cost < target rate"


def test_decision_rule_arm_selection():
    """
    Verify Stage D:
    a_t = argmax [ hat{r}_{t,a}(x_t) - lambda_t * hat{c}_{t,a}(x_t) ]
    """
    policy = DiscountedLinCBwK(
        n_arms=3,
        context_dim=4,
        initial_budget=1.0,
        total_horizon=10,
        alpha=0.1,
        initial_lambda=2.0,
    )
    x = np.array([1.0, 0.0, 0.0, 0.0])

    # Manually configure arms:
    # Arm 0: High reward (0.9), High cost (0.4) -> Score = 0.9 - 2.0 * 0.4 = 0.10
    # Arm 1: Moderate reward (0.7), Low cost (0.05) -> Score = 0.7 - 2.0 * 0.05 = 0.60
    # Arm 2: Low reward (0.2), Low cost (0.05) -> Score = 0.2 - 2.0 * 0.05 = 0.10
    policy.theta[0, 0] = 0.9
    policy.psi[0, 0] = 0.4

    policy.theta[1, 0] = 0.7
    policy.psi[1, 0] = 0.05

    policy.theta[2, 0] = 0.2
    policy.psi[2, 0] = 0.05

    sel: ArmSelectionResult = policy.select_arm(x)
    assert sel.arm_id == 1, "Arm 1 should dominate due to gas penalty on Arm 0"
    assert sel.all_scores[1] > sel.all_scores[0]


def test_stopping_conditions():
    """
    Verify Section 4 stopping conditions:
      1. Horizon completed (t >= T)
      2. Budget exhausted (B_t <= 0)
      3. Budget below min arm cost
    """
    # 1. Horizon completed
    p1 = DiscountedLinCBwK(n_arms=2, context_dim=3, total_horizon=2, initial_budget=10.0)
    x = np.ones(3) / math.sqrt(3)
    p1.update(0, x, 0.5, cost=0.1)
    p1.update(0, x, 0.5, cost=0.1)
    stopped, reason = p1.check_stopping_condition(x)
    assert stopped is True
    assert reason == "horizon_completed"

    # 2. Budget exhausted
    p2 = DiscountedLinCBwK(n_arms=2, context_dim=3, total_horizon=100, initial_budget=0.10)
    p2.update(0, x, 0.5, cost=0.15)
    stopped, reason = p2.check_stopping_condition(x)
    assert stopped is True
    assert reason == "budget_exhausted"

    # 3. Budget below min arm cost
    p3 = DiscountedLinCBwK(n_arms=2, context_dim=3, total_horizon=100, initial_budget=0.08)
    p3.psi[0, 0] = 0.20
    p3.psi[1, 0] = 0.20
    stopped, reason = p3.check_stopping_condition(x)
    assert stopped is True
    assert reason == "budget_below_min_arm_cost"


def test_checkpoint_roundtrip():
    """Verify state serialization to and from JSON."""
    policy = DiscountedLinCBwK(
        n_arms=3,
        context_dim=5,
        initial_budget=0.75,
        total_horizon=30,
        gamma=0.88,
        alpha=0.18,
    )
    x = np.array([1.0, 0.5, -0.2, 0.1, 0.0])
    policy.update(arm_id=2, context=x, reward=0.65, cost=0.04)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "d_lincbwk_ckpt.json"
        policy.save_checkpoint(ckpt_path)

        loaded = DiscountedLinCBwK(n_arms=3, context_dim=5)
        loaded.load_checkpoint(ckpt_path)

        assert loaded.total_rounds == 1
        assert loaded.arm_counts[2] == 1
        assert math.isclose(loaded.cumulative_reward, 0.65)
        assert math.isclose(loaded.cumulative_cost, 0.04)
        np.testing.assert_allclose(loaded.theta, policy.theta)
        np.testing.assert_allclose(loaded.psi, policy.psi)
        np.testing.assert_allclose(loaded.A_inv, policy.A_inv)


def test_alias_compatibility():
    assert DiscountedLinUCBWithKnapsack is DiscountedLinCBwK


def test_persamaan_3_16_and_3_19_exact_alignment():
    """
    Verify Chapter 3 Persamaan (3.16) clipping [0, 1] and Persamaan (3.19)
    ridge floor A_{t+1,a_t} = gamma * A_{t,a_t} + x_t x_t^T + (1 - gamma) * lambda_reg * I_d,
    while unpulled arms remain unchanged (Algoritma 3.1 lines 30-31).
    """
    d = 4
    gamma = 0.82
    l2_reg = 1.5
    policy = DiscountedLinCBwK(
        n_arms=2,
        context_dim=d,
        gamma=gamma,
        alpha=2.0,
        l2_lambda=l2_reg,
    )
    x = np.array([1.0, 0.5, 0.2, 0.1], dtype=np.float64)

    # Check Persamaan (3.16): UCB reward estimate is clipped to [0.0, 1.0]
    _, _, ucb_r = policy.estimate_reward(0, x)
    assert 0.0 <= ucb_r <= 1.0
    assert math.isclose(ucb_r, 1.0)

    A0_before = policy.A[0].copy()
    A1_before = policy.A[1].copy()

    policy.update(arm_id=0, context=x, reward=1.0, cost=0.01)

    expected_A0 = (
        gamma * A0_before
        + np.outer(x, x)
        + (1.0 - gamma) * l2_reg * np.eye(d)
    )
    np.testing.assert_allclose(policy.A[0], expected_A0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(policy.A_inv[0], np.linalg.inv(expected_A0), rtol=1e-6, atol=1e-6)

    # Unpulled arm 1 must remain untouched
    np.testing.assert_allclose(policy.A[1], A1_before, rtol=1e-6, atol=1e-6)


