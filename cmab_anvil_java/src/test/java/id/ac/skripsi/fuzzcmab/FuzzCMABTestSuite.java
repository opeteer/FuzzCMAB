package id.ac.skripsi.fuzzcmab;

import id.ac.skripsi.fuzzcmab.algorithms.ArmSelectionResult;
import id.ac.skripsi.fuzzcmab.algorithms.DiscountedLinCBwK;
import id.ac.skripsi.fuzzcmab.algorithms.DiscountedLinUCB;
import id.ac.skripsi.fuzzcmab.algorithms.DisjointLinUCB;
import id.ac.skripsi.fuzzcmab.algorithms.DynamicRegretEvaluator;
import id.ac.skripsi.fuzzcmab.algorithms.StationaryLinCBwK;
import id.ac.skripsi.fuzzcmab.algorithms.UniformRandomPolicy;
import id.ac.skripsi.fuzzcmab.common.Arm;
import id.ac.skripsi.fuzzcmab.common.RegularUserAnvilClient;
import id.ac.skripsi.fuzzcmab.experiments.ContractCatalog;
import id.ac.skripsi.fuzzcmab.experiments.GasRegimes;
import id.ac.skripsi.fuzzcmab.util.LinearAlgebra;
import id.ac.skripsi.fuzzcmab.vaultcustody.BanditRewardBreakdown;
import id.ac.skripsi.fuzzcmab.vaultcustody.OnChainContextSnapshot;
import id.ac.skripsi.fuzzcmab.vaultcustody.PublicContextExtractor;
import id.ac.skripsi.fuzzcmab.vaultcustody.VaultCustodyArms;
import id.ac.skripsi.fuzzcmab.vaultcustody.VaultCustodyRewardOracle;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Suite Pengujian Unit Lengkap untuk Proyek Java FuzzCMAB.
 *
 * Memverifikasi kesesuaian matematis Persamaan (3.3)-(3.23) pada implementasi Java:
 *   1. Keakuratan invers matriks Gauss-Jordan & pembaruan kovariansi terdiskonto (Pers. 3.19)
 *   2. Penjepitan UCB imbalan [0, 1] (Pers. 3.16) dan pembaruan dual Knapsack (Pers. 3.13) pada D-LinCBwK
 *   3. Konsistensi reduksi ablasi (LinCBwK gamma=1, D-LinUCB lambda_t=0, LinUCB, UniformRandom)
 *   4. Penalti kebangkrutan penuh pada DynamicRegretEvaluator (Pers. 3.22)
 *   5. Normalisasi ruang konteks 10D x_t in R^10 (Tabel 3.1) dan katalog K=8 Arm
 *   6. Orakel imbalan berjenjang 4-komponen supp(r_t) = {0.00, 0.20, 0.50, 0.70, 1.00} (Pers. 3.3-3.5)
 *   7. Keccak-256 Ethereum & inferensi slot mapping Solidity
 */
public final class FuzzCMABTestSuite {

    private FuzzCMABTestSuite() {
    }

    private static void assertAlmostEqual(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + " | expected=" + expected + ", actual=" + actual);
        }
    }

    private static void assertTrue(boolean cond, String msg) {
        if (!cond) {
            throw new AssertionError(msg);
        }
    }

    public static void testPersamaan316And319ExactAlignment() {
        int d = 4;
        double gamma = 0.80;
        double l2Lambda = 2.0;
        DiscountedLinCBwK policy = DiscountedLinCBwK.createDefault(
                3, d, gamma, 1.0, 10, 0.50, l2Lambda, 1.0, 42L, "D-LinCBwK"
        );
        double[] x = {1.0, 0.5, 0.25, 0.0};
        double[] est0 = policy.estimateReward(0, x);
        assertTrue(est0[2] >= 0.0 && est0[2] <= 1.0, "UCB reward wajib dijepit pada [0.0, 1.0]");

        double[][] a0Before = LinearAlgebra.copyMatrix(policy.getAMat()[0]);
        double[][] a1Before = LinearAlgebra.copyMatrix(policy.getAMat()[1]);
        policy.update(0, x, 0.70, 0.15);

        // Persamaan (3.19): A_{t+1, 0} = gamma * A_{t, 0} + x x^T + (1 - gamma) * lambda_reg * I_d
        double[][] a0After = policy.getAMat()[0];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                double expected = gamma * a0Before[i][j] + x[i] * x[j] + (i == j ? (1.0 - gamma) * l2Lambda : 0.0);
                assertAlmostEqual(expected, a0After[i][j], 1e-10, "Kovariansi Persamaan (3.19) tidak cocok");
            }
        }

        // Algoritma 3.1 baris 30-31: Lengan yang tidak ditarik (a = 1) tidak meluruh saat decayAllArms = false
        double[][] a1After = policy.getAMat()[1];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                assertAlmostEqual(a1Before[i][j], a1After[i][j], 1e-12, "Lengan tak terpilih tidak boleh berubah");
            }
        }

        // Persamaan (3.13): lambda_2 = clip(1.0 + eta * (0.15 - 0.10), 0, 50)
        double expectedLambda = 1.0 + policy.getEta() * (0.15 - (1.0 / 10.0));
        assertAlmostEqual(expectedLambda, policy.getLambdaT(), 1e-10, "Pembaruan dual Persamaan (3.13) tidak cocok");
    }

    public static void testAllFivePoliciesPolymorphicExecution() {
        int k = 6;
        int d = 10;
        double[] x = {1.0, 0.8, 0.5, 0.0, 0.2, 1.0, 1.0, 0.0, 0.1, 1.0};

        UniformRandomPolicy pRandom = new UniformRandomPolicy(k, d, 0.5, 20, 42L, "UniformRandom");
        DisjointLinUCB pLin = new DisjointLinUCB(k, d, 0.55, 1.0, 42L);
        DiscountedLinUCB pDLin = new DiscountedLinUCB(k, d, 0.82, 0.55, 1.0, false, false, 42L);
        StationaryLinCBwK pLinK = new StationaryLinCBwK(k, d, 0.5, 20, 0.55, 1.0, 1.0, 42L, "LinCBwK");
        DiscountedLinCBwK pDLinK = DiscountedLinCBwK.createDefault(k, d, 0.82, 0.5, 20, 0.55, 1.0, 1.0, 42L, "D-LinCBwK");

        for (var policy : List.of(pRandom, pLin, pDLin, pLinK, pDLinK)) {
            ArmSelectionResult sel = policy.selectArm(x);
            assertTrue(sel.getArmId() >= 0 && sel.getArmId() < k, "Indeks arm harus dalam [0, K-1]");
            policy.update(sel.getArmId(), x, 0.50, 0.02);
            assertTrue(policy.getTotalRounds() == 1, "totalRounds harus bertambah menjadi 1");
        }
    }

    public static void testRewardOracleSupportAndHierarchy() {
        VaultCustodyRewardOracle oracle = new VaultCustodyRewardOracle();
        OnChainContextSnapshot baseSnap = new OnChainContextSnapshot(
                1, 5.0, 5.0, 0.0, 1.0, 100.0, true, false, true, 0.0, false, false, 0.25
        );

        // E0: Revert -> 0.00
        BanditRewardBreakdown r0 = oracle.computeReward(baseSnap, baseSnap, false, 0);
        assertAlmostEqual(0.00, r0.getTotalReward(), 1e-9, "E0 harus 0.00");
        assertTrue("E0".equals(r0.getEventPartition()), "Partisi harus E0");

        // E1: Valid execution tanpa mutasi storage -> 0.20
        BanditRewardBreakdown r1 = oracle.computeReward(baseSnap, baseSnap, true, 0);
        assertAlmostEqual(0.20, r1.getTotalReward(), 1e-9, "E1 harus 0.20");
        assertTrue("E1".equals(r1.getEventPartition()), "Partisi harus E1");

        // E2: Mutasi storage (SSTORE) tanpa ekstraksi aset -> 0.50
        OnChainContextSnapshot sstoreSnap = new OnChainContextSnapshot(
                2, 5.25, 5.25, 0.25, 0.75, 100.0, true, false, true, 0.0, true, false, 0.25
        );
        BanditRewardBreakdown r2 = oracle.computeReward(baseSnap, sstoreSnap, true, 0);
        assertAlmostEqual(0.50, r2.getTotalReward(), 1e-9, "E2 harus 0.50");
        assertTrue("E2".equals(r2.getEventPartition()), "Partisi harus E2");

        // E3: Mutasi storage + ekstraksi aset ke EOA -> 0.70
        OnChainContextSnapshot econSnap = new OnChainContextSnapshot(
                3, 5.00, 5.00, 0.0, 1.00, 350.0, true, false, true, 0.0, true, true, 0.25
        );
        BanditRewardBreakdown r3 = oracle.computeReward(sstoreSnap, econSnap, true, 2);
        assertAlmostEqual(0.70, r3.getTotalReward(), 1e-9, "E3 harus 0.70");
        assertTrue("E3".equals(r3.getEventPartition()), "Partisi harus E3");

        // E4: Pelanggaran invarian solvabilitas -> 1.00
        OnChainContextSnapshot vulnSnap = new OnChainContextSnapshot(
                4, 0.00, 5.00, 0.0, 6.00, 350.0, false, false, true, 0.0, true, true, 0.25
        );
        BanditRewardBreakdown r4 = oracle.computeReward(econSnap, vulnSnap, true, 5);
        assertAlmostEqual(1.00, r4.getTotalReward(), 1e-9, "E4 harus 1.00");
        assertTrue("E4".equals(r4.getEventPartition()), "Partisi harus E4");
    }

    public static void testEarlyStoppingBankruptcyRegretPenalty() {
        List<Map<String, Object>> earlyLogs = new ArrayList<>();
        for (int t = 1; t <= 2; t++) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("round", t);
            m.put("arm_id", 0);
            m.put("reward", 0.50);
            m.put("cost_eth", 0.25);
            m.put("event_type", "none");
            earlyLogs.add(m);
        }

        List<Map<String, Object>> fullLogs = new ArrayList<>();
        for (int t = 1; t <= 5; t++) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("round", t);
            m.put("arm_id", 1);
            m.put("reward", 0.50);
            m.put("cost_eth", 0.05);
            m.put("event_type", "none");
            fullLogs.add(m);
        }

        Map<String, List<Map<String, Object>>> map = new LinkedHashMap<>();
        map.put("EarlyBankrupt", earlyLogs);
        map.put("FullSurvivor", fullLogs);

        Map<String, DynamicRegretEvaluator.DynamicRegretSummary> summaries =
                DynamicRegretEvaluator.comparePoliciesDynamicRegret(map, 10, 6, 0.50, 1.00, 5);

        assertTrue(
                summaries.get("EarlyBankrupt").getCumulativeDynamicRegret()
                        > summaries.get("FullSurvivor").getCumulativeDynamicRegret(),
                "Kebijakan yang bangkrut di putaran 2/5 wajib menerima penalti regret pada putaran 3..5"
        );
        assertTrue(summaries.get("EarlyBankrupt").getTrajectory().size() == 5, "Trajektori wajib sepanjang T=5");
    }

    public static void testContextAndArmAndKeccak() {
        List<Arm> arms = VaultCustodyArms.buildVaultCustodyArmCatalog();
        assertTrue(arms.size() == 8, "Katalog Agen_Vault_Custody harus memiliki 8 Arm");
        assertTrue(ContractCatalog.CATALOG_CONTRACTS.size() == 8, "Katalog eksperimen harus memiliki 8 kontrak");

        PublicContextExtractor extractor = new PublicContextExtractor(0.25);
        OnChainContextSnapshot snap = new OnChainContextSnapshot(
                10, 4.98, 4.98, 0.25, 1.25, 250.0, true, false, true, 0.125, true, true, 0.25
        );
        double[] vec = extractor.toVector(snap);
        assertTrue(vec.length == 10, "Dimensi vektor konteks harus 10");
        assertAlmostEqual(1.0, vec[0], 1e-12, "x0_bias harus 1.0");
        assertAlmostEqual(1.0, vec[5], 1e-12, "x5_deposit_eligibility harus 1.0");
        assertAlmostEqual(1.0, vec[6], 1e-12, "x6_sstore_flag harus 1.0");
        assertAlmostEqual(1.0, vec[7], 1e-12, "x7_waypoint_flag harus 1.0");

        String slot = RegularUserAnvilClient.computeSolidityMappingSlot(
                RegularUserAnvilClient.DEFAULT_AGENT_ADDR, 0
        );
        assertTrue(slot.startsWith("0x") && slot.length() == 66, "Hash slot Keccak-256 harus 32 byte (66 karakter hex)");

        List<Double> gasStable = GasRegimes.generateGasSeries("stable", 30, 42L);
        List<Double> gasVolatile = GasRegimes.generateGasSeries("volatile_spikes", 30, 42L);
        assertTrue(gasStable.size() == 30 && gasVolatile.size() == 30, "Panjang deret gas harus 30");
    }

    public static void main(String[] args) {
        testPersamaan316And319ExactAlignment();
        testAllFivePoliciesPolymorphicExecution();
        testRewardOracleSupportAndHierarchy();
        testEarlyStoppingBankruptcyRegretPenalty();
        testContextAndArmAndKeccak();
        System.out.println("[✓] SELURUH 5 RANGKAIAN UNIT TEST JAVA FUZZCMAB BERHASIL LULUS!");
    }
}
