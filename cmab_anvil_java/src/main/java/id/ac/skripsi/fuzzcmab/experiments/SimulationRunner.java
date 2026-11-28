package id.ac.skripsi.fuzzcmab.experiments;

import id.ac.skripsi.fuzzcmab.algorithms.ArmSelectionResult;
import id.ac.skripsi.fuzzcmab.common.Arm;
import id.ac.skripsi.fuzzcmab.common.StepTransition;
import id.ac.skripsi.fuzzcmab.util.SimpleJson;
import id.ac.skripsi.fuzzcmab.vaultcustody.PublicContextExtractor;
import id.ac.skripsi.fuzzcmab.vaultcustody.VaultCustodyCMABAgent;
import id.ac.skripsi.fuzzcmab.vaultcustody.VaultCustodyEnv;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * CLI Simulation Runner untuk Agen_Vault_Custody di Anvil (Porting dari run_cmab_simulation.py).
 */
public final class SimulationRunner {

    private SimulationRunner() {
    }

    public static void main(String[] args) throws Exception {
        String rpcUrl = ExperimentConfig.RPC_URL;
        String algorithm = "d-lincbwk";
        int rounds = 25;
        double budget = 1.50;
        String gasRegime = "stable";
        double gamma = ExperimentConfig.GAMMA;
        double alpha = ExperimentConfig.ALPHA;
        double l2Lambda = ExperimentConfig.L2_LAMBDA;
        boolean theoreticalBeta = false;
        boolean decayAllArms = false;
        boolean peerTraffic = true;
        Path outputLog = ExperimentConfig.RESULTS_DIR.resolve("cmab_simulation_results.json");

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--rpc-url" -> rpcUrl = args[++i];
                case "--algorithm" -> algorithm = args[++i];
                case "--rounds" -> rounds = Integer.parseInt(args[++i]);
                case "--budget" -> budget = Double.parseDouble(args[++i]);
                case "--gas-regime" -> gasRegime = args[++i];
                case "--gamma" -> gamma = Double.parseDouble(args[++i]);
                case "--alpha" -> alpha = Double.parseDouble(args[++i]);
                case "--l2-lambda" -> l2Lambda = Double.parseDouble(args[++i]);
                case "--theoretical-beta" -> theoreticalBeta = true;
                case "--decay-all-arms" -> decayAllArms = true;
                case "--no-peer-traffic" -> peerTraffic = false;
                case "--output-log" -> outputLog = Paths.get(args[++i]);
                default -> {
                }
            }
        }

        System.out.println("=".repeat(96));
        System.out.printf("SIMULASI CMAB JAVA (%s) - AGEN_VAULT_CUSTODY PADA ANVIL%n", algorithm.toUpperCase());
        System.out.println("=".repeat(96));

        VaultCustodyEnv env = new VaultCustodyEnv(
                rpcUrl,
                null,
                budget,
                gasRegime,
                rounds,
                peerTraffic,
                ExperimentConfig.MIN_CAPITAL_THRESHOLD_ETH,
                42L
        );

        Map<String, String> standing = env.verifyRegularUserStanding();
        System.out.println("[*] Verifikasi Kedudukan Agen Pengguna Biasa (Regular User Standing Check):");
        System.out.println("    - Spesifikasi Agen   : " + standing.get("agent_name"));
        System.out.println("    - Alamat EOA Agen    : " + standing.get("agent_address") + " (Pengguna Biasa)");
        System.out.println("    - Owner Kontrak Vault: " + standing.get("vault_owner") + " (Bukan Agen)");
        System.out.println("    - Alamat Vault       : " + standing.get("vault_address"));
        System.out.println("    - Alamat MockToken   : " + standing.get("token_address"));
        System.out.println("    - Status Kedudukan   : " + standing.get("standing"));
        System.out.printf(Locale.US, "    - Rezim Harga Gas    : %s | Anggaran Modal B = %.2f ETH%n", gasRegime, budget);
        System.out.println("-".repeat(96));

        VaultCustodyCMABAgent agent = new VaultCustodyCMABAgent(
                env,
                algorithm,
                alpha,
                l2Lambda,
                gamma,
                budget,
                rounds,
                theoreticalBeta,
                decayAllArms,
                42L
        );

        List<Map<String, Object>> history = new ArrayList<>();

        for (int t = 1; t <= rounds; t++) {
            if (env.getRemainingBudgetEth() <= env.getMinCapitalThresholdEth()) {
                System.out.printf(Locale.US,
                        "[!] Anggaran modal gas habis pada putaran ke-%d (B = %.4f ETH).%n",
                        t, env.getRemainingBudgetEth());
                break;
            }

            VaultCustodyCMABAgent.ActionTransitionPair pair = agent.actAndLearn();
            ArmSelectionResult sel = pair.getSelection();
            StepTransition tr = pair.getTransition();
            double reward = tr.getRewardBreakdown().getTotalReward();
            String statusTag = tr.isTxSucceeded() ? "SUCCESS" : "REVERT ";

            System.out.printf(Locale.US,
                    "[Round %02d/%02d] Gas=%5.1fGwei | Ext=%-19s | Arm #%d (%-27s) | Tx=%s | Rew=%.2f (%s) | Cost=%.4f | B_t=%.4f | VaultETH=%.2f%n",
                    t, rounds,
                    tr.getGasPriceGwei(),
                    tr.getExternalEvent(),
                    sel.getArmId(),
                    tr.getArmSpec().getName(),
                    statusTag,
                    reward,
                    tr.getRewardBreakdown().getEventPartition(),
                    tr.getTxCostEth(),
                    tr.getRemainingBudgetEth(),
                    tr.getNextSnapshot().getVaultEthBalance()
            );

            for (String anomaly : tr.getRewardBreakdown().getDetectedAnomalies()) {
                System.out.println("    [!] TEMUAN ANOMALI / KERENTANAN: " + anomaly);
            }

            Map<String, Object> rec = new LinkedHashMap<>();
            rec.put("round", t);
            rec.put("gas_price_gwei", tr.getGasPriceGwei());
            rec.put("external_event", tr.getExternalEvent());
            rec.put("arm_id", sel.getArmId());
            rec.put("arm_name", tr.getArmSpec().getName());
            rec.put("ucb_score", sel.getUcbScore());
            rec.put("expected_reward", sel.getExpectedReward());
            rec.put("uncertainty_bonus", sel.getUncertaintyBonus());
            rec.put("tx_succeeded", tr.isTxSucceeded());
            rec.put("reward", reward);
            rec.put("cost_eth", tr.getTxCostEth());
            rec.put("remaining_budget_eth", tr.getRemainingBudgetEth());
            rec.put("reward_breakdown", tr.getRewardBreakdown().toMap());
            rec.put("context", tr.getContextVector());
            rec.put("vault_eth_after", tr.getNextSnapshot().getVaultEthBalance());
            rec.put("agent_eoa_eth_after", tr.getNextSnapshot().getAgentEoaEthBalance());
            rec.put("agent_vault_stake_after", tr.getNextSnapshot().getAgentVaultBalance());
            rec.put("vault_solvent_after", tr.getNextSnapshot().isVaultIsSolvent());
            history.add(rec);
        }

        Map<String, Object> diag = agent.getPolicy().getDiagnostics();
        System.out.println("-".repeat(96));
        System.out.printf("RINGKASAN EVALUASI AGEN_VAULT_CUSTODY (%s):%n", agent.getPolicy().getName());
        System.out.printf("    - Total Putaran Bertahan  : %s/%d%n", diag.get("total_rounds"), rounds);
        System.out.printf(Locale.US, "    - Akumulasi Reward        : %+.4f%n", ((Number) diag.get("cumulative_reward")).doubleValue());
        System.out.printf(Locale.US, "    - Rata-rata Reward / Tx   : %+.4f%n", ((Number) diag.get("mean_reward")).doubleValue());
        System.out.printf(Locale.US, "    - Sisa Anggaran Gas (B_T) : %.4f ETH%n", env.getRemainingBudgetEth());
        System.out.printf("    - Total Anomali Terdeteksi: %d%n", agent.getDiscoveredVulnerabilities().size());
        long[] counts = agent.getPolicy().getArmCounts();
        for (Arm arm : env.getArms()) {
            System.out.printf("        [%d] %-28s: %3dx dipilih%n", arm.getArmId(), arm.getName(), counts[arm.getArmId()]);
        }
        System.out.println("=".repeat(96));

        Files.createDirectories(outputLog.getParent());
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("standing", standing);
        payload.put("diagnostics", diag);
        payload.put("discovered_vulnerabilities", agent.getDiscoveredVulnerabilities());
        payload.put("feature_names", PublicContextExtractor.FEATURE_NAMES);
        payload.put("history", history);
        Files.writeString(outputLog, SimpleJson.toPrettyJson(payload));
    }
}
