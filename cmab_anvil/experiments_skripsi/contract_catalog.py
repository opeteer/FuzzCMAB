"""
Katalog Kontrak SC-Bench Berdasarkan Kedalaman State (Kondisi 1 Eksperimen Skripsi).

Mengkategorisasikan kontrak target ke dalam 3 tier kompleksitas state EVM:
  - Tier 1: Shallow State (Mutasi linier sederhana, 1-2 SSTORE per tx)
  - Tier 2: Medium State (Prasyarat berurutan, escrow settlement, fee toggles)
  - Tier 3: Deep Multi-State (Dependensi waktu block.timestamp, multi-tier staking, Merkle proof)
"""

from typing import Dict, List, Any, Optional

DEPLOYER_ADDR = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
PEER_ADDR = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
AGENT_ADDR = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"

TIER_METADATA: Dict[str, Dict[str, Any]] = {
    "benchmark": {
        "tier_id": "TIER_0_BENCHMARK",
        "name": "Controlled Synthetic Benchmark (VulnerableVault)",
        "description": "Kontrak tolok ukur sintetis terkontrol dengan prasyarat likuiditas ambang 5 ETH dan selfdestruct/insolvency oracle.",
        "expected_sstore_range": "2 - 3 slots",
    },
    "shallow": {
        "tier_id": "TIER_1_SHALLOW",
        "name": "Shallow State (Kompleksitas Rendah)",
        "description": "Mutasi linier langsung, 1-2 slot storage termodifikasi per aksi, tanpa prasyarat berantai.",
        "expected_sstore_range": "1 - 2 slots",
    },
    "medium": {
        "tier_id": "TIER_2_MEDIUM",
        "name": "Medium State (Kompleksitas Sedang)",
        "description": "Status berurutan multi-tahap (deposit -> balance update -> approval -> settlement).",
        "expected_sstore_range": "3 - 4 slots",
    },
    "deep": {
        "tier_id": "TIER_3_DEEP",
        "name": "Deep Multi-State (Kompleksitas Tinggi)",
        "description": "Dependensi waktu on-chain, multi-order array, reward multipliers, referral tree, dan Merkle verification.",
        "expected_sstore_range": ">= 5 slots",
    },
}

CATALOG_CONTRACTS: List[Dict[str, Any]] = [
    # =========================================================================
    # TIER 0: CONTROLLED SYNTHETIC BENCHMARK (Sub-bab 3.4a)
    # =========================================================================
    {
        "id": "SKRIPSI-B01",
        "contract_idx": 0,
        "tier": "benchmark",
        "split": "benchmark",
        "file": "VulnerableVault.sol",
        "contract": "VulnerableVault",
        "category": "Controlled_Synthetic_Vault",
        "description": "Kontrak tolok ukur sintetis terkontrol dengan prasyarat saldo >= 5 ETH pada emergencyWithdraw().",
        "arms": [
            {"id": 0, "name": "vault_deposit_small", "sig": "deposit()", "args": [], "val": 0.02, "role": "inflow_primary"},
            {"id": 1, "name": "vault_deposit_medium", "sig": "deposit()", "args": [], "val": 0.05, "role": "inflow_secondary"},
            {"id": 2, "name": "vault_withdraw_stake", "sig": "withdraw()", "args": [], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "vault_emergency_withdraw", "sig": "emergencyWithdraw()", "args": [], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "vault_zero_deposit_probe", "sig": "deposit()", "args": [], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "vault_eth_fallback", "sig": "", "args": [], "val": 0.01, "role": "fallback_ops"},
        ],
    },

    # =========================================================================
    # TIER 1: SHALLOW STATE (Sub-bab 3.4b)
    # =========================================================================
    {
        "id": "SKRIPSI-S01",
        "contract_idx": 1,
        "tier": "shallow",
        "split": "injected",
        "file": "ClashForStash-0x939b0425.sol",
        "contract": "ClashForStash",
        "category": "Vault_Custody_Yield",
        "description": "Token custody vault dengan min/max deposit caps dan uninitialized trustedSigner (address(0)) ecrecover bypass.",
        "arms": [
            {"id": 0, "name": "member_deposit_small", "sig": "deposit(uint256)", "args": ["100000000000000000"], "val": 0.0, "role": "inflow_primary"},
            {"id": 1, "name": "member_deposit_medium", "sig": "deposit(uint256)", "args": ["150000000000000000"], "val": 0.0, "role": "inflow_secondary"},
            {"id": 2, "name": "member_transfer_peer", "sig": "transfer(address,uint256)", "args": [PEER_ADDR, "50000000000000000"], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_increase_allowance", "sig": "increaseAllowance(address,uint256)", "args": [PEER_ADDR, "100000000000000000"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "unsigned_withdraw_probe", "sig": "withdraw(uint256,bytes)", "args": ["250000000000000000", "DYNAMIC_SIG65"], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "vault_eth_fallback", "sig": "", "args": [], "val": 0.05, "role": "fallback_ops"},
        ],
    },
    {
        "id": "SKRIPSI-S02",
        "contract_idx": 2,
        "tier": "shallow",
        "split": "injected",
        "file": "Token-0x7c81418a.sol",
        "contract": "Token",
        "category": "Liquidity_Vault_Token",
        "description": "Taxed liquidity token vault dengan unprotected public ETH flush (withdraw()) dan fee-bearing transfers.",
        "arms": [
            {"id": 0, "name": "member_transfer_small", "sig": "transfer(address,uint256)", "args": [PEER_ADDR, "100000000000000"], "val": 0.0, "role": "inflow_primary"},
            {"id": 1, "name": "member_transfer_medium", "sig": "transfer(address,uint256)", "args": [DEPLOYER_ADDR, "200000000000000"], "val": 0.0, "role": "inflow_secondary"},
            {"id": 2, "name": "public_withdraw_tax_eth", "sig": "withdraw()", "args": [], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_approve_spender", "sig": "approve(address,uint256)", "args": [PEER_ADDR, "500000000000000"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "admin_removeLimits_probe", "sig": "removeLimits()", "args": [], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "vault_eth_fallback", "sig": "", "args": [], "val": 0.03, "role": "fallback_ops"},
        ],
    },

    # =========================================================================
    # TIER 2: MEDIUM STATE
    # =========================================================================
    {
        "id": "SKRIPSI-M01",
        "contract_idx": 3,
        "tier": "medium",
        "split": "injected",
        "file": "DrawTT-0x02D33D35.sol",
        "contract": "DrawTT",
        "constructor_sig": "constructor(string,string,string)",
        "constructor_args": ["DrawTTCustody", "DTT", "ipfs://vault/"],
        "category": "Escrow_PullPayment_Vault",
        "description": "Custodial NFT & PullPayment escrow contract dengan dynamic mint fee dan async settlement.",
        "arms": [
            {"id": 0, "name": "member_mint_DynamicURI", "sig": "mint(string)", "args": ["DYNAMIC_URI"], "val": 0.02, "role": "inflow_primary"},
            {"id": 1, "name": "member_mint_lowfee", "sig": "mint(string)", "args": ["DYNAMIC_URI_LOW"], "val": 0.01, "role": "inflow_secondary"},
            {"id": 2, "name": "escrow_withdraw_payments", "sig": "withdrawPayments(address)", "args": [DEPLOYER_ADDR], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_set_approval_all", "sig": "setApprovalForAll(address,bool)", "args": [PEER_ADDR, "true"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "unauthorized_setCost_probe", "sig": "setCost(uint256)", "args": ["0"], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "duplicate_uri_probe", "sig": "mint(string)", "args": ["STATIC_DUPLICATE"], "val": 0.02, "role": "fallback_ops"},
        ],
    },
    {
        "id": "SKRIPSI-M02",
        "contract_idx": 4,
        "tier": "medium",
        "split": "origin",
        "file": "Shuffle-0x137110Eb.sol",
        "contract": "Shuffle",
        "category": "Liquidity_Pool_Lottery",
        "description": "Custodial bidding room pool dengan time-locked bid withdrawal dan unprotected modifyRoom hijack.",
        "arms": [
            {"id": 0, "name": "member_place_bid_room1", "sig": "placeBid(uint256)", "args": ["1"], "val": 0.05, "role": "inflow_primary"},
            {"id": 1, "name": "member_place_bid_room2", "sig": "placeBid(uint256)", "args": ["2"], "val": 0.08, "role": "inflow_secondary"},
            {"id": 2, "name": "member_withdraw_bid_room1", "sig": "withdrawBid(uint256)", "args": ["1"], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "unprotected_modify_room2", "sig": "modifyRoom(uint256,bool,uint256,address)", "args": ["2", "true", "90", AGENT_ADDR], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "admin_createRoom_probe", "sig": "createRoom(string,uint256,uint256,uint256,uint256,uint256,address)", "args": ["Hack", "100", "1000", "1", "5", "60", AGENT_ADDR], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "zero_bid_probe", "sig": "placeBid(uint256)", "args": ["1"], "val": 0.0, "role": "fallback_ops"},
        ],
    },

    # =========================================================================
    # TIER 3: DEEP MULTI-STATE
    # =========================================================================
    {
        "id": "SKRIPSI-D01",
        "contract_idx": 5,
        "tier": "deep",
        "split": "injected",
        "file": "PyramidToken-0xbc69f076.sol",
        "contract": "PyramidToken",
        "category": "Liquidity_Pyramid_Vault",
        "description": "Tiered staking & liquidity vault dengan 30-day lockup maturity, unbacked referral bonus, dan burn mechanics.",
        "arms": [
            {"id": 0, "name": "member_stake_30days", "sig": "stake(uint256,uint256)", "args": ["1000000000000000000", "2592000"], "val": 0.0, "role": "inflow_primary"},
            {"id": 1, "name": "member_set_referrer", "sig": "setReferrer(address)", "args": [PEER_ADDR], "val": 0.0, "role": "inflow_secondary"},
            {"id": 2, "name": "member_withdraw_mature", "sig": "withdraw()", "args": [], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_burn_tokens", "sig": "burn(uint256)", "args": ["100000000000000000"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "member_transfer_peer", "sig": "transfer(address,uint256)", "args": [PEER_ADDR, "100000000000000000"], "val": 0.0, "role": "fallback_ops"},
            {"id": 5, "name": "invalid_short_stake_probe", "sig": "stake(uint256,uint256)", "args": ["100000000000000000", "60"], "val": 0.0, "role": "probe"},
        ],
    },
    {
        "id": "SKRIPSI-D02",
        "contract_idx": 6,
        "tier": "deep",
        "split": "origin",
        "file": "PymeStaking-0x90b8fbb7.sol",
        "contract": "PymeStaking",
        "category": "TimeLocked_Staking_Custody",
        "description": "Upgradeable staking custody dengan lockup tiers, high-yield reserve drain, dan claim gates.",
        "arms": [
            {"id": 0, "name": "member_stake_tier0", "sig": "stake(uint256,uint256)", "args": ["500000000000000000", "1"], "val": 0.0, "role": "inflow_primary"},
            {"id": 1, "name": "member_stake_tier1", "sig": "stake(uint256,uint256)", "args": ["1000000000000000000", "7"], "val": 0.0, "role": "inflow_secondary"},
            {"id": 2, "name": "member_claim_mature", "sig": "claim(uint256)", "args": ["DYNAMIC_STAKE_IDX"], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_unlock_premature", "sig": "unlockPremature(uint256)", "args": ["DYNAMIC_STAKE_IDX"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "admin_withdraw_probe", "sig": "withdraw()", "args": [], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "admin_setReward_probe", "sig": "setReward(uint256,uint256)", "args": ["1", "5000"], "val": 0.0, "role": "fallback_ops"},
        ],
    },
    {
        "id": "SKRIPSI-D03",
        "contract_idx": 7,
        "tier": "deep",
        "split": "origin",
        "file": "Airdrop-0x897b328c.sol",
        "contract": "Airdrop",
        "category": "Custodial_Merkle_Airdrop",
        "description": "Merkle-verified token distribution custody dengan single-node root claim dan token underlying operations.",
        "arms": [
            {"id": 0, "name": "member_merkle_claim_100", "sig": "claim(uint256,bytes32[])", "args": ["100000000000000000000", "[]"], "val": 0.0, "role": "inflow_primary"},
            {"id": 1, "name": "member_merkle_claim_250", "sig": "claim(uint256,bytes32[])", "args": ["250000000000000000000", "[]"], "val": 0.0, "role": "inflow_secondary"},
            {"id": 2, "name": "member_token_transfer", "target": "MOCK_TOKEN", "sig": "transfer(address,uint256)", "args": [PEER_ADDR, "10000000000000000000"], "val": 0.0, "role": "secondary_ops"},
            {"id": 3, "name": "member_token_approve", "target": "MOCK_TOKEN", "sig": "approve(address,uint256)", "args": [PEER_ADDR, "50000000000000000000"], "val": 0.0, "role": "emergency_ops"},
            {"id": 4, "name": "admin_setRoot_probe", "sig": "setRoot(bytes32)", "args": ["0x0000000000000000000000000000000000000000000000000000000000000000"], "val": 0.0, "role": "probe"},
            {"id": 5, "name": "admin_setToken_probe", "sig": "setToken(address)", "args": [AGENT_ADDR], "val": 0.0, "role": "fallback_ops"},
        ],
    },
]


def get_contracts_by_tier(tier: str) -> List[Dict[str, Any]]:
    """Mengembalikan daftar spesifikasi kontrak untuk tier tertentu ('benchmark', 'shallow', 'medium', 'deep')."""
    tier_clean = tier.lower().strip()
    matched = [c for c in CATALOG_CONTRACTS if c["tier"] == tier_clean]
    if not matched:
        raise ValueError(
            f"Tier tidak ditemukan: '{tier}'. Pilih 'benchmark', 'shallow', 'medium', atau 'deep'."
        )
    return matched


def get_all_catalog_contracts() -> List[Dict[str, Any]]:
    """Mengembalikan seluruh 7 kontrak representatif skripsi."""
    return CATALOG_CONTRACTS

