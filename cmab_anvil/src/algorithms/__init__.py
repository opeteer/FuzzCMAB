"""
Algorithm templates and policies for Contextual Multi-Armed Bandits (CMAB).
"""

from .base import BaseCMABPolicy, ArmSelectionResult
from .linucb import LinUCBTemplate, DisjointLinUCB, HybridLinUCBTemplate
from .discounted_linucb import DiscountedLinUCBTemplate, DiscountedLinUCB
from .discounted_linucb_knapsack import DiscountedLinCBwK, DiscountedLinUCBWithKnapsack
from .linucb_knapsack import (
    LinUCBwK,
    LinUCBWithKnapsack,
    DisjointLinUCBWithKnapsack,
    StationaryLinCBwK,
)
from .uniform_random import UniformRandomPolicy
from .dynamic_regret import (
    DynamicRegretEvaluator,
    DynamicRegretStep,
    DynamicRegretSummary,
    compare_policies_dynamic_regret,
)

__all__ = [
    "BaseCMABPolicy",
    "ArmSelectionResult",
    "UniformRandomPolicy",
    "LinUCBTemplate",
    "DisjointLinUCB",
    "HybridLinUCBTemplate",
    "DiscountedLinUCBTemplate",
    "DiscountedLinUCB",
    "DiscountedLinCBwK",
    "DiscountedLinUCBWithKnapsack",
    "LinUCBwK",
    "LinUCBWithKnapsack",
    "DisjointLinUCBWithKnapsack",
    "StationaryLinCBwK",
    "DynamicRegretEvaluator",
    "DynamicRegretStep",
    "DynamicRegretSummary",
    "compare_policies_dynamic_regret",
]
