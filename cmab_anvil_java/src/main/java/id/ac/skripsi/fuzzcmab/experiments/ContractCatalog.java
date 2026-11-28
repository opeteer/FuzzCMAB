package id.ac.skripsi.fuzzcmab.experiments;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Katalog Kontrak Eksperimen Skripsi Berdasarkan Kedalaman State (Kondisi 1 - Sub-bab 3.4):
 *   - Tier 0 (benchmark): VulnerableVault (SKRIPSI-B01)
 *   - Tier 1 (shallow)  : ClashForStash (SKRIPSI-S01), Token (SKRIPSI-S02)
 *   - Tier 2 (medium)   : DrawTT (SKRIPSI-M01), Shuffle (SKRIPSI-M02)
 *   - Tier 3 (deep)     : PyramidToken (SKRIPSI-D01), PymeStaking (SKRIPSI-D02), Airdrop (SKRIPSI-D03)
 */
public final class ContractCatalog {

    public static final class CatalogArmSpec {
        private final int id;
        private final String name;
        private final String target;
        private final String sig;
        private final List<String> args;
        private final double val;
        private final String role;

        public CatalogArmSpec(
                int id,
                String name,
                String target,
                String sig,
                List<String> args,
                double val,
                String role
        ) {
            this.id = id;
            this.name = name;
            this.target = target;
            this.sig = sig != null ? sig : "";
            this.args = args != null ? Collections.unmodifiableList(new ArrayList<>(args)) : Collections.emptyList();
            this.val = val;
            this.role = role != null ? role : "";
        }

        public static CatalogArmSpec of(int id, String name, String sig, List<String> args, double val, String role) {
            return new CatalogArmSpec(id, name, null, sig, args, val, role);
        }

        public int getId() {
            return id;
        }

        public String getName() {
            return name;
        }

        public String getTarget() {
            return target;
        }

        public String getSig() {
            return sig;
        }

        public List<String> getArgs() {
            return args;
        }

        public double getVal() {
            return val;
        }

        public String getRole() {
            return role;
        }
    }

    public static final class ContractSpec {
        private final String id;
        private final int contractIdx;
        private final String tier;
        private final String split;
        private final String file;
        private final String contract;
        private final String constructorSig;
        private final List<String> constructorArgs;
        private final String category;
        private final String description;
        private final List<CatalogArmSpec> arms;

        public ContractSpec(
                String id,
                int contractIdx,
                String tier,
                String split,
                String file,
                String contract,
                String constructorSig,
                List<String> constructorArgs,
                String category,
                String description,
                List<CatalogArmSpec> arms
        ) {
            this.id = id;
            this.contractIdx = contractIdx;
            this.tier = tier;
            this.split = split;
            this.file = file;
            this.contract = contract;
            this.constructorSig = constructorSig;
            this.constructorArgs = constructorArgs != null
                    ? Collections.unmodifiableList(new ArrayList<>(constructorArgs))
                    : Collections.emptyList();
            this.category = category;
            this.description = description;
            this.arms = Collections.unmodifiableList(new ArrayList<>(arms));
        }

        public String getId() {
            return id;
        }

        public int getContractIdx() {
            return contractIdx;
        }

        public String getTier() {
            return tier;
        }

        public String getSplit() {
            return split;
        }

        public String getFile() {
            return file;
        }

        public String getContract() {
            return contract;
        }

        public String getConstructorSig() {
            return constructorSig;
        }

        public List<String> getConstructorArgs() {
            return constructorArgs;
        }

        public String getCategory() {
            return category;
        }

        public String getDescription() {
            return description;
        }

        public List<CatalogArmSpec> getArms() {
            return arms;
        }
    }

    private static final String DEPLOYER_ADDR = ExperimentConfig.DEPLOYER_ADDR;
    private static final String PEER_ADDR = ExperimentConfig.PEER_ADDR;
    private static final String AGENT_ADDR = ExperimentConfig.AGENT_ADDR;

    public static final List<ContractSpec> CATALOG_CONTRACTS = List.of(
            // TIER 0: BENCHMARK (VulnerableVault)
            new ContractSpec(
                    "SKRIPSI-B01", 0, "benchmark", "benchmark",
                    "VulnerableVault.sol", "VulnerableVault", null, null,
                    "Controlled_Synthetic_Vault",
                    "Kontrak tolok ukur sintetis terkontrol dengan prasyarat saldo >= 5 ETH pada emergencyWithdraw().",
                    List.of(
                            CatalogArmSpec.of(0, "vault_deposit_small", "deposit()", List.of(), 0.02, "inflow_primary"),
                            CatalogArmSpec.of(1, "vault_deposit_medium", "deposit()", List.of(), 0.05, "inflow_secondary"),
                            CatalogArmSpec.of(2, "vault_withdraw_stake", "withdraw()", List.of(), 0.0, "secondary_ops"),
                            CatalogArmSpec.of(3, "vault_emergency_withdraw", "emergencyWithdraw()", List.of(), 0.0, "emergency_ops"),
                            CatalogArmSpec.of(4, "vault_zero_deposit_probe", "deposit()", List.of(), 0.0, "probe"),
                            CatalogArmSpec.of(5, "vault_eth_fallback", "", List.of(), 0.01, "fallback_ops")
                    )
            ),
            // TIER 1: SHALLOW STATE
            new ContractSpec(
                    "SKRIPSI-S01", 1, "shallow", "injected",
                    "ClashForStash-0x939b0425.sol", "ClashForStash", null, null,
                    "Vault_Custody_Yield",
                    "Token custody vault dengan min/max deposit caps dan uninitialized trustedSigner (address(0)) ecrecover bypass.",
                    List.of(
                            CatalogArmSpec.of(0, "member_deposit_small", "deposit(uint256)", List.of("100000000000000000"), 0.0, "inflow_primary"),
                            CatalogArmSpec.of(1, "member_deposit_medium", "deposit(uint256)", List.of("150000000000000000"), 0.0, "inflow_secondary"),
                            CatalogArmSpec.of(2, "member_transfer_peer", "transfer(address,uint256)", List.of(PEER_ADDR, "50000000000000000"), 0.0, "secondary_ops"),
                            CatalogArmSpec.of(3, "member_increase_allowance", "increaseAllowance(address,uint256)", List.of(PEER_ADDR, "100000000000000000"), 0.0, "emergency_ops"),
                            CatalogArmSpec.of(4, "unsigned_withdraw_probe", "withdraw(uint256,bytes)", List.of("250000000000000000", "DYNAMIC_SIG65"), 0.0, "probe"),
                            CatalogArmSpec.of(5, "vault_eth_fallback", "", List.of(), 0.05, "fallback_ops")
                    )
            ),
            new ContractSpec(
                    "SKRIPSI-S02", 2, "shallow", "injected",
                    "Token-0x7c81418a.sol", "Token", null, null,
                    "Liquidity_Vault_Token",
                    "Taxed liquidity token vault dengan unprotected public ETH flush (withdraw()) dan fee-bearing transfers.",
                    List.of(
                            CatalogArmSpec.of(0, "member_transfer_small", "transfer(address,uint256)", List.of(PEER_ADDR, "100000000000000"), 0.0, "inflow_primary"),
                            CatalogArmSpec.of(1, "member_transfer_medium", "transfer(address,uint256)", List.of(DEPLOYER_ADDR, "200000000000000"), 0.0, "inflow_secondary"),
                            CatalogArmSpec.of(2, "public_withdraw_tax_eth", "withdraw()", List.of(), 0.0, "secondary_ops"),
                            CatalogArmSpec.of(3, "member_approve_spender", "approve(address,uint256)", List.of(PEER_ADDR, "500000000000000"), 0.0, "emergency_ops"),
                            CatalogArmSpec.of(4, "admin_removeLimits_probe", "removeLimits()", List.of(), 0.0, "probe"),
                            CatalogArmSpec.of(5, "vault_eth_fallback", "", List.of(), 0.03, "fallback_ops")
                    )
            ),
            // TIER 2: MEDIUM STATE
            new ContractSpec(
                    "SKRIPSI-M01", 3, "medium", "injected",
                    "DrawTT-0x02D33D35.sol", "DrawTT",
                    "constructor(string,string,string)", List.of("DrawTTCustody", "DTT", "ipfs://vault/"),
                    "Escrow_PullPayment_Vault",
                    "Custodial NFT & PullPayment escrow contract dengan dynamic mint fee dan async settlement.",
                    List.of(
                            CatalogArmSpec.of(0, "member_mint_DynamicURI", "mint(string)", List.of("DYNAMIC_URI"), 0.02, "inflow_primary"),
                            CatalogArmSpec.of(1, "member_mint_lowfee", "mint(string)", List.of("DYNAMIC_URI_LOW"), 0.01, "inflow_secondary"),
                            CatalogArmSpec.of(2, "escrow_withdraw_payments", "withdrawPayments(address)", List.of(DEPLOYER_ADDR), 0.0, "secondary_ops"),
                            CatalogArmSpec.of(3, "member_set_approval_all", "setApprovalForAll(address,bool)", List.of(PEER_ADDR, "true"), 0.0, "emergency_ops"),
                            CatalogArmSpec.of(4, "unauthorized_setCost_probe", "setCost(uint256)", List.of("0"), 0.0, "probe"),
                            CatalogArmSpec.of(5, "duplicate_uri_probe", "mint(string)", List.of("STATIC_DUPLICATE"), 0.02, "fallback_ops")
                    )
            ),
            new ContractSpec(
                    "SKRIPSI-M02", 4, "medium", "origin",
                    "Shuffle-0x137110Eb.sol", "Shuffle", null, null,
                    "Liquidity_Pool_Lottery",
                    "Custodial bidding room pool dengan time-locked bid withdrawal dan unprotected modifyRoom hijack.",
                    List.of(
                            CatalogArmSpec.of(0, "member_place_bid_room1", "placeBid(uint256)", List.of("1"), 0.05, "inflow_primary"),
                            CatalogArmSpec.of(1, "member_place_bid_room2", "placeBid(uint256)", List.of("2"), 0.08, "inflow_secondary"),
                            CatalogArmSpec.of(2, "member_withdraw_bid_room1", "withdrawBid(uint256)", List.of("1"), 0.0, "secondary_ops"),
                            CatalogArmSpec.of(3, "unprotected_modify_room2", "modifyRoom(uint256,bool,uint256,address)", List.of("2", "true", "90", AGENT_ADDR), 0.0, "emergency_ops"),
                            CatalogArmSpec.of(4, "admin_createRoom_probe", "createRoom(string,uint256,uint256,uint256,uint256,uint256,address)", List.of("Hack", "100", "1000", "1", "5", "60", AGENT_ADDR), 0.0, "probe"),
                            CatalogArmSpec.of(5, "zero_bid_probe", "placeBid(uint256)", List.of("1"), 0.0, "fallback_ops")
                    )
            ),
            // TIER 3: DEEP MULTI-STATE
            new ContractSpec(
                    "SKRIPSI-D01", 5, "deep", "injected",
                    "PyramidToken-0xbc69f076.sol", "PyramidToken", null, null,
                    "Liquidity_Pyramid_Vault",
                    "Tiered staking & liquidity vault dengan 30-day lockup maturity, unbacked referral bonus, dan burn mechanics.",
                    List.of(
                            CatalogArmSpec.of(0, "member_stake_30days", "stake(uint256,uint256)", List.of("1000000000000000000", "2592000"), 0.0, "inflow_primary"),
                            CatalogArmSpec.of(1, "member_set_referrer", "setReferrer(address)", List.of(PEER_ADDR), 0.0, "inflow_secondary"),
                            CatalogArmSpec.of(2, "member_withdraw_mature", "withdraw()", List.of(), 0.0, "secondary_ops"),
                            CatalogArmSpec.of(3, "member_burn_tokens", "burn(uint256)", List.of("100000000000000000"), 0.0, "emergency_ops"),
                            CatalogArmSpec.of(4, "member_transfer_peer", "transfer(address,uint256)", List.of(PEER_ADDR, "100000000000000000"), 0.0, "fallback_ops"),
                            CatalogArmSpec.of(5, "invalid_short_stake_probe", "stake(uint256,uint256)", List.of("100000000000000000", "60"), 0.0, "probe")
                    )
            ),
            new ContractSpec(
                    "SKRIPSI-D02", 6, "deep", "origin",
                    "PymeStaking-0x90b8fbb7.sol", "PymeStaking", null, null,
                    "TimeLocked_Staking_Custody",
                    "Upgradeable staking custody dengan lockup tiers, high-yield reserve drain, dan claim gates.",
                    List.of(
                            CatalogArmSpec.of(0, "member_stake_tier0", "stake(uint256,uint256)", List.of("500000000000000000", "1"), 0.0, "inflow_primary"),
                            CatalogArmSpec.of(1, "member_stake_tier1", "stake(uint256,uint256)", List.of("1000000000000000000", "7"), 0.0, "inflow_secondary"),
                            CatalogArmSpec.of(2, "member_claim_mature", "claim(uint256)", List.of("DYNAMIC_STAKE_IDX"), 0.0, "secondary_ops"),
                            CatalogArmSpec.of(3, "member_unlock_premature", "unlockPremature(uint256)", List.of("DYNAMIC_STAKE_IDX"), 0.0, "emergency_ops"),
                            CatalogArmSpec.of(4, "admin_withdraw_probe", "withdraw()", List.of(), 0.0, "probe"),
                            CatalogArmSpec.of(5, "admin_setReward_probe", "setReward(uint256,uint256)", List.of("1", "5000"), 0.0, "fallback_ops")
                    )
            ),
            new ContractSpec(
                    "SKRIPSI-D03", 7, "deep", "origin",
                    "Airdrop-0x897b328c.sol", "Airdrop", null, null,
                    "Custodial_Merkle_Airdrop",
                    "Merkle-verified token distribution custody dengan single-node root claim dan token underlying operations.",
                    List.of(
                            CatalogArmSpec.of(0, "member_merkle_claim_100", "claim(uint256,bytes32[])", List.of("100000000000000000000", "[]"), 0.0, "inflow_primary"),
                            CatalogArmSpec.of(1, "member_merkle_claim_250", "claim(uint256,bytes32[])", List.of("250000000000000000000", "[]"), 0.0, "inflow_secondary"),
                            new CatalogArmSpec(2, "member_token_transfer", "MOCK_TOKEN", "transfer(address,uint256)", List.of(PEER_ADDR, "10000000000000000000"), 0.0, "secondary_ops"),
                            new CatalogArmSpec(3, "member_token_approve", "MOCK_TOKEN", "approve(address,uint256)", List.of(PEER_ADDR, "50000000000000000000"), 0.0, "emergency_ops"),
                            CatalogArmSpec.of(4, "admin_setRoot_probe", "setRoot(bytes32)", List.of("0x0000000000000000000000000000000000000000000000000000000000000000"), 0.0, "probe"),
                            CatalogArmSpec.of(5, "admin_setToken_probe", "setToken(address)", List.of(AGENT_ADDR), 0.0, "fallback_ops")
                    )
            )
    );

    public static List<ContractSpec> getContractsByTier(String tier) {
        String clean = tier != null ? tier.toLowerCase().trim() : "all";
        if ("all".equals(clean)) {
            return CATALOG_CONTRACTS;
        }
        List<ContractSpec> matched = new ArrayList<>();
        for (ContractSpec c : CATALOG_CONTRACTS) {
            if (c.getTier().equals(clean)) {
                matched.add(c);
            }
        }
        if (matched.isEmpty()) {
            throw new IllegalArgumentException(
                    "Tier tidak ditemukan: '" + tier + "'. Pilih 'benchmark', 'shallow', 'medium', atau 'deep'."
            );
        }
        return matched;
    }
}
