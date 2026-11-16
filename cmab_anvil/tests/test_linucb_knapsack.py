"""
Unit test suite for Standard LinUCB with Knapsacks (LinUCBwK / Stationary).
"""

import math
import tempfile
from pathlib import Path
import numpy as np
import pytest

from cmab_anvil.algorithms.linucb_knapsack import (
    LinUCBwK,
    LinUCBWithKnapsack,
    DisjointLinUCBWithKnapsack,
    StationaryLinCBwK,
)
from cmab_anvil.algorithms.base import ArmSelectionResult


def test_linucbwk_initialization():
    policy = LinUCBwK(
        n_arms=4,
        context_dim=10,
        initial_budget=0.60,
        total_horizon=24,
        alpha=0.15,
        l2_lambda=1.0,
        initial_lambda=1.0,
    )
    assert policy.n_arms == 4
    assert policy.context_dim == 10
    assert policy.remaining_budget == 0.60
    assert policy.lambda_t == 1.0
    assert policy.total_rounds == 0
    assert math.isclose(policy.eta, math.sqrt(math.log(4) / 24), rel_tol=1e-5)
    assert np.allclose(policy.A[0], np.eye(10))
    assert np.allclose(policy.A_inv[0], np.eye(10))


def test_linucbwk_accumulators_and_estimation():
    K, d = 3, 4
    policy = LinUCBwK(n_arms=K, context_dim=d, alpha=0.2, l2_lambda=1.0)
    x = np.array([1.0, 0.5, 0.2, 0.1], dtype=np.float64)

    # Initial estimates with prior (theta=0, psi=0, A_inv=I)
    mean_r, unc_r, ucb_r = policy.estimate_reward(0, x)
    assert mean_r == 0.0
    expected_unc = 0.2 * math.sqrt(float(np.dot(x, x)))
    assert math.isclose(unc_r, expected_unc, rel_tol=1e-5)
    assert math.isclose(ucb_r, expected_unc, rel_tol=1e-5)

    exp_c, pred_c = policy.estimate_cost(0, x)
    assert exp_c == 0.0
    assert pred_c == policy.min_cost  # Lower bounded by 0.001


def test_linucbwk_sherman_morrison_accuracy():
    """
    Verify that stationary rank-1 Sherman-Morrison update matches exact np.linalg.inv.
    """
    d = 5
    policy = LinUCBwK(n_arms=2, context_dim=d, l2_lambda=1.5, use_sherman_morrison=True)

    rng = np.random.default_rng(42)
    x = rng.standard_normal(d)

    A_old = policy.A[0].copy()
    policy.update(arm_id=0, context=x, reward=0.7, cost=0.03)

    # Ground truth: (A_old + x x^T)^(-1)
    A_exact = A_old + np.outer(x, x)
    A_inv_exact = np.linalg.inv(A_exact)

    np.testing.assert_allclose(policy.A_inv[0], A_inv_exact, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(policy.A[0], A_exact, rtol=1e-6, atol=1e-6)


def test_linucbwk_stationary_accumulation():
    """
    Verify that observations accumulate without memory discounting (gamma = 1.0),
    and unpulled arms remain completely untouched.
    """
    policy = LinUCBwK(n_arms=2, context_dim=3, l2_lambda=1.0)
    x1 = np.array([1.0, 0.0, 0.0])
    x2 = np.array([0.0, 1.0, 0.0])

    policy.update(arm_id=0, context=x1, reward=0.6, cost=0.02)
    policy.update(arm_id=0, context=x2, reward=0.4, cost=0.03)

    # Pulled arm 0 accumulates sum(x x^T), sum(r x), sum(c x)
    expected_b_r = 0.6 * x1 + 0.4 * x2
    expected_b_c = 0.02 * x1 + 0.03 * x2
    np.testing.assert_allclose(policy.b_reward[0], expected_b_r)
    np.testing.assert_allclose(policy.b_cost[0], expected_b_c)

    # Unpulled arm 1 remains at initial prior
    np.testing.assert_allclose(policy.A[1], np.eye(3))
    np.testing.assert_allclose(policy.b_reward[1], np.zeros(3))
    np.testing.assert_allclose(policy.b_cost[1], np.zeros(3))


def test_linucbwk_exponential_dual_update():
    """
    Verify Exponential Dual Update:
      lambda_t = lambda_{t-1} * exp( eta_t * (cost - B/T) )
    """
    B, T = 1.0, 10
    target_quota = B / T  # 0.10
    policy = LinUCBwK(
        n_arms=2,
        context_dim=3,
        initial_budget=B,
        total_horizon=T,
        initial_lambda=1.0,
        dual_update_mode="observed",
    )
    x = np.array([1.0, 0.5, 0.2])

    # Expensive transaction: cost = 0.25 > 0.10 -> lambda increases
    policy.update(arm_id=0, context=x, reward=0.8, cost=0.25)
    assert policy.lambda_t > 1.0
    assert math.isclose(policy.remaining_budget, 0.75)

    lambda_after_exp = policy.lambda_t

    # Frugal transaction: cost = 0.02 < 0.10 -> lambda decreases
    policy.update(arm_id=0, context=x, reward=0.6, cost=0.02)
    assert policy.lambda_t < lambda_after_exp


def test_linucbwk_decision_rule():
    """
    Verify a_t = argmax [ r_hat - lambda_t * c_hat ]
    """
    policy = LinUCBwK(
        n_arms=3,
        context_dim=3,
        initial_budget=1.0,
        total_horizon=10,
        alpha=0.1,
        initial_lambda=2.0,
    )
    x = np.array([1.0, 0.0, 0.0])

    # Arm 0: High reward 0.95, High cost 0.40 -> Score: 0.95 - 2.0*0.40 = 0.15
    # Arm 1: Good reward 0.70, Low cost 0.05  -> Score: 0.70 - 2.0*0.05 = 0.60
    # Arm 2: Low reward 0.20, Low cost 0.05   -> Score: 0.20 - 2.0*0.05 = 0.10
    policy.theta[0, 0] = 0.95
    policy.psi[0, 0] = 0.40

    policy.theta[1, 0] = 0.70
    policy.psi[1, 0] = 0.05

    policy.theta[2, 0] = 0.20
    policy.psi[2, 0] = 0.05

    sel: ArmSelectionResult = policy.select_arm(x)
    assert sel.arm_id == 1, "Arm 1 should be selected due to gas penalty on Arm 0"
    assert sel.all_scores[1] > sel.all_scores[0]


def test_linucbwk_stopping_conditions():
    # 1. Horizon completed
    p1 = LinUCBwK(n_arms=2, context_dim=2, total_horizon=2, initial_budget=5.0)
    x = np.array([1.0, 0.0])
    p1.update(0, x, 0.5, cost=0.1)
    p1.update(0, x, 0.5, cost=0.1)
    stopped, reason = p1.check_stopping_condition(x)
    assert stopped is True
    assert reason == "horizon_completed"

    # 2. Budget exhausted
    p2 = LinUCBwK(n_arms=2, context_dim=2, total_horizon=100, initial_budget=0.10)
    p2.update(0, x, 0.5, cost=0.12)
    stopped, reason = p2.check_stopping_condition(x)
    assert stopped is True
    assert reason == "budget_exhausted"

    # 3. Budget below min arm cost
    p3 = LinUCBwK(n_arms=2, context_dim=2, total_horizon=100, initial_budget=0.05)
    p3.psi[0, 0] = 0.15
    p3.psi[1, 0] = 0.20
    stopped, reason = p3.check_stopping_condition(x)
    assert stopped is True
    assert reason == "budget_below_min_arm_cost"


def test_linucbwk_checkpoint_roundtrip():
    policy = LinUCBwK(
        n_arms=3,
        context_dim=4,
        initial_budget=0.80,
        total_horizon=25,
        alpha=0.16,
    )
    x = np.array([1.0, 0.4, -0.1, 0.2])
    policy.update(arm_id=1, context=x, reward=0.6, cost=0.035)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "linucbwk_ckpt.json"
        policy.save_checkpoint(ckpt_path)

        loaded = LinUCBwK(n_arms=3, context_dim=4)
        loaded.load_checkpoint(ckpt_path)

        assert loaded.total_rounds == 1
        assert loaded.arm_counts[1] == 1
        assert math.isclose(loaded.cumulative_reward, 0.6)
        assert math.isclose(loaded.cumulative_cost, 0.035)
        np.testing.assert_allclose(loaded.theta, policy.theta)
        np.testing.assert_allclose(loaded.psi, policy.psi)
        np.testing.assert_allclose(loaded.A_inv, policy.A_inv)


def test_linucbwk_aliases():
    assert LinUCBWithKnapsack is LinUCBwK
    assert DisjointLinUCBWithKnapsack is LinUCBwK
    assert StationaryLinCBwK is LinUCBwK

