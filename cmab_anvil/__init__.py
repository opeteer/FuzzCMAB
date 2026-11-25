"""
FuzzCMAB: Contextual Multi-Armed Bandits with Knapsacks (D-LinCBwK) Framework
for Unprivileged Smart Contract Vulnerability Fuzzing on Anvil.

Directory Structure:
  - `src/`                 : Core algorithms, RPC client, and `Agen_Vault_Custody` domain agent
  - `experiments_skripsi/` : Multi-condition factorial benchmark suite & publication plots
  - `tests/`               : Unit tests (`test_*.py`) and simulation runner scripts
  - `datasets/`            : Curated SC-Bench dataset & categorization tools
  - `results/`             : Simulation telemetry, checkpoints, reports, and charts
"""

import sys
from .src import algorithms, common, Agen_Vault_Custody
from .src.algorithms import (
    BaseCMABPolicy,
    ArmSelectionResult,
    UniformRandomPolicy,
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
    Arm,
    BaseUnprivilegedBanditEnv,
    StepTransition,
)
from .src.Agen_Vault_Custody import (
    OnChainContextSnapshot,
    PublicContextExtractor,
    CONTEXT_DIM,
    FEATURE_NAMES,
    build_vault_custody_arm_catalog,
    BanditRewardBreakdown,
    VaultCustodyRewardOracle,
    REWARD_SUPPORT_LEVELS,
    REWARD_WEIGHT_VECTOR,
    VaultCustodyEnv,
    VaultCustodyCMABAgent,
    generate_vault_gas_series,
    generate_external_party_events,
)

# Register backward-compatibility aliases so `cmab_anvil.algorithms`, etc. resolve directly
sys.modules[f"{__name__}.algorithms"] = algorithms
sys.modules[f"{__name__}.common"] = common
sys.modules[f"{__name__}.Agen_Vault_Custody"] = Agen_Vault_Custody

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
    "UnprivilegedAnvilClient",
    "Arm",
    "BaseUnprivilegedBanditEnv",
    "StepTransition",
    "OnChainContextSnapshot",
    "PublicContextExtractor",
    "CONTEXT_DIM",
    "FEATURE_NAMES",
    "build_vault_custody_arm_catalog",
    "BanditRewardBreakdown",
    "VaultCustodyRewardOracle",
    "REWARD_SUPPORT_LEVELS",
    "REWARD_WEIGHT_VECTOR",
    "VaultCustodyEnv",
    "VaultCustodyCMABAgent",
    "generate_vault_gas_series",
    "generate_external_party_events",
    "algorithms",
    "common",
    "Agen_Vault_Custody",
]
