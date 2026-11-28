package id.ac.skripsi.fuzzcmab.experiments;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Konfigurasi Terpusat Eksperimen Skripsi (Porting dari experiments_skripsi/config.py).
 */
public final class ExperimentConfig {

    private ExperimentConfig() {
    }

    public static final Path REPO_ROOT = resolveRepoRoot();
    public static final Path JAVA_PROJECT_ROOT = REPO_ROOT.resolve("cmab_anvil_java");
    public static final Path RESULTS_DIR = JAVA_PROJECT_ROOT.resolve("results");
    public static final Path TELEMETRY_DIR = RESULTS_DIR.resolve("telemetry");

    public static final Path DATASET_DIR = resolveDatasetDir();
    public static final Path MOCK_TOKEN_SOL = REPO_ROOT.resolve("anvil_env").resolve("src").resolve("MockToken.sol");
    public static final Path VULNERABLE_VAULT_SOL = REPO_ROOT.resolve("anvil_env").resolve("src").resolve("VulnerableVault.sol");

    public static final String SOLC_BIN = "/home/opeteer/.local/bin/solc";
    public static final String CAST_BIN = "/home/opeteer/.foundry/bin/cast";
    public static final String ANVIL_BIN = "/home/opeteer/.foundry/bin/anvil";
    public static final String RPC_URL = "http://127.0.0.1:8545";

    public static final String DEPLOYER_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
    public static final String DEPLOYER_ADDR = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266";

    public static final String PEER_PK = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";
    public static final String PEER_ADDR = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8";

    public static final String AGENT_PK = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a";
    public static final String AGENT_ADDR = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC";

    public static final int CONTEXT_DIM = 10;
    public static final int DEFAULT_HORIZON_T = 50;

    public static final double ALPHA = 0.55;
    public static final double GAMMA = 0.82;
    public static final double L2_LAMBDA = 1.0;
    public static final double INITIAL_LAMBDA = 1.0;
    public static final double MIN_CAPITAL_THRESHOLD_ETH = 0.005;

    public static final List<String> POLICY_NAMES = List.of(
            "UniformRandom",
            "LinUCB",
            "D-LinUCB",
            "LinCBwK",
            "D-LinCBwK"
    );

    public static final Map<String, Double> BUDGET_TIERS = createBudgetTiers();

    private static Map<String, Double> createBudgetTiers() {
        Map<String, Double> map = new LinkedHashMap<>();
        map.put("tight", 0.05);
        map.put("moderate", 0.15);
        map.put("loose", 0.50);
        return map;
    }

    private static Path resolveRepoRoot() {
        Path curr = Paths.get("").toAbsolutePath();
        for (Path p = curr; p != null; p = p.getParent()) {
            if (Files.exists(p.resolve("anvil_env"))) {
                return p;
            }
        }
        return curr;
    }

    private static Path resolveDatasetDir() {
        Path candidate1 = REPO_ROOT.resolve("cmab_anvil")
                .resolve("datasets").resolve("SC-Bench")
                .resolve("categorized_dataset").resolve("Vault_and_Liquidity");
        if (Files.exists(candidate1)) {
            return candidate1;
        }
        return REPO_ROOT.resolve("SC-Bench")
                .resolve("categorized_dataset").resolve("Vault_and_Liquidity");
    }
}
