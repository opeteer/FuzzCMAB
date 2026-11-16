"""
Contextual Multi-Armed Bandits (CMAB) Framework for Unprivileged Blockchain Agents on Anvil.

Directory Structure:
  - `src/`      : Core source code (algorithms, common, Agen_Vault_Custody)
  - `tests/`    : Simulation and experiment runner scripts
  - `datasets/` : Datasets (SC-Bench and categorization scripts)
  - `results/`  : Simulation outputs, checkpoints, reports, and charts
"""

import sys
from .src import algorithms, common, Agen_Vault_Custody
from .src.algorithms import (
    BaseCMABPolicy,
    ArmSelectionResult,
    LinUCBTemplate,
    DisjointLinUCB,
    HybridLinUCBTemplate,
    DiscountedLinUCBTemplate,
    DiscountedLinUCB,
    DiscountedLinCBwK,
    DiscountedLinUCBWithKnapsack,
    LinUCBwK,
    LinUCBWithKnapsack,
    DisjointLinUCBWithKnapsack,
    StationaryLinCBwK,
    DynamicRegretEvaluator,
    DynamicRegretStep,
    DynamicRegretSummary,
    compare_policies_dynamic_regret,
)
from .src.common import (
    UnprivilegedAnvilClient,
    BaseArmSpec,
    BaseUnprivilegedBanditEnv,
    StepTransition,
)
from .src.Agen_Vault_Custody import (
    VaultCustodyEnv,
    VaultCustodyCMABAgent,
    CONTEXT_DIM,
    FEATURE_NAMES,
)

# Register backward-compatibility aliases so `cmab_anvil.algorithms`, etc. resolve directly
sys.modules[f"{__name__}.algorithms"] = algorithms
sys.modules[f"{__name__}.common"] = common
sys.modules[f"{__name__}.Agen_Vault_Custody"] = Agen_Vault_Custody

__all__ = [
    "BaseCMABPolicy",
    "ArmSelectionResult",
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
    "UnprivilegedAnvilClient",
    "BaseArmSpec",
    "BaseUnprivilegedBanditEnv",
    "StepTransition",
    "VaultCustodyEnv",
    "VaultCustodyCMABAgent",
    "CONTEXT_DIM",
    "FEATURE_NAMES",
    "algorithms",
    "common",
    "Agen_Vault_Custody",
]
