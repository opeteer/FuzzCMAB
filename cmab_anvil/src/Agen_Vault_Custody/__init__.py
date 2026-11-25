"""
Agen_Vault_Custody: Specialized Unprivileged CMAB Agent for Custody Vault & ERC-20 Contracts
(`VulnerableVault.sol` and `MockToken.sol`) on the Anvil Simulator.
"""

from .context import (
    OnChainContextSnapshot,
    PublicContextExtractor,
    CONTEXT_DIM,
    FEATURE_NAMES,
)
from .arms import build_vault_custody_arm_catalog
from .reward import (
    BanditRewardBreakdown,
    VaultCustodyRewardOracle,
    REWARD_SUPPORT_LEVELS,
    REWARD_WEIGHT_VECTOR,
)
from .agent import (
    VaultCustodyEnv,
    VaultCustodyCMABAgent,
    generate_vault_gas_series,
    generate_external_party_events,
)

__all__ = [
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
]

