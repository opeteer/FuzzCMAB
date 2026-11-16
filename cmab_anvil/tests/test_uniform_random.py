"""
Unit tests for UniformRandomPolicy baseline (Forge Fuzz Invariant Proxy).
"""

import math
from pathlib import Path
import tempfile
import numpy as np
import pytest

from cmab_anvil.algorithms.uniform_random import UniformRandomPolicy


def test_uniform_random_initialization():
    policy = UniformRandomPolicy(n_arms=6, context_dim=10, initial_budget=1.0, total_horizon=24)
    assert policy.n_arms == 6
    assert policy.context_dim == 10
    assert policy.remaining_budget == 1.0
    assert policy.total_rounds == 0
    assert policy.cumulative_reward == 0.0
    assert policy.cumulative_cost == 0.0
    assert len(policy.arm_counts) == 6


def test_uniform_random_arm_selection_distribution():
    K, d = 4, 5
    policy = UniformRandomPolicy(n_arms=K, context_dim=d, seed=123)
    x = np.ones(d)

    selections = [policy.select_arm(x).arm_id for _ in range(1000)]
    counts = np.bincount(selections, minlength=K)

    # In uniform distribution with 1000 trials, each arm should get around 250 pulls (+- 60)
    for c in counts:
        assert 180 <= c <= 320, f"Arm count {c} deviated too much from expected 250"


def test_uniform_random_update_and_budget_tracking():
    policy = UniformRandomPolicy(n_arms=3, context_dim=4, initial_budget=0.50, total_horizon=10)
    x = np.zeros(4)

    # Update with reward and cost
    policy.update(arm_id=1, context=x, reward=0.8, cost=0.05)
    assert policy.total_rounds == 1
    assert policy.arm_counts[1] == 1
    assert math.isclose(policy.cumulative_reward, 0.8)
    assert math.isclose(policy.cumulative_cost, 0.05)
    assert math.isclose(policy.remaining_budget, 0.45)
    assert math.isclose(policy.theta[1, 0], 0.8)


def test_uniform_random_stopping_conditions():
    policy = UniformRandomPolicy(n_arms=2, context_dim=3, initial_budget=0.10, total_horizon=3)
    x = np.ones(3)

    # Round 1
    policy.update(0, x, 0.5, cost=0.04)
    stopped, reason = policy.check_stopping_condition(x)
    assert not stopped
    assert reason == "continue"

    # Round 2 - budget exhausted
    policy.update(0, x, 0.5, cost=0.08)
    stopped, reason = policy.check_stopping_condition(x)
    assert stopped
    assert reason == "budget_exhausted"


def test_uniform_random_checkpoint_roundtrip():
    policy = UniformRandomPolicy(n_arms=3, context_dim=4, initial_budget=1.0, total_horizon=20)
    x = np.array([1.0, 0.0, 0.5, -0.2])
    policy.update(2, x, 0.75, cost=0.12)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "random_ckpt.json"
        policy.save_checkpoint(ckpt_path)

        loaded = UniformRandomPolicy(n_arms=3, context_dim=4)
        loaded.load_checkpoint(ckpt_path)

        assert loaded.total_rounds == 1
        assert loaded.arm_counts[2] == 1
        assert math.isclose(loaded.cumulative_reward, 0.75)
        assert math.isclose(loaded.cumulative_cost, 0.12)

