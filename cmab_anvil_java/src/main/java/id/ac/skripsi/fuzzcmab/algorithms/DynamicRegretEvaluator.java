package id.ac.skripsi.fuzzcmab.algorithms;

import id.ac.skripsi.fuzzcmab.util.LinearAlgebra;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Evaluator Cumulative Dynamic Regret D-Regret(T) sesuai Persamaan (3.22) dan Sub-bab 3.6 Butir 1:
 *
 *     D-Regret(T) = sum_{t=1}^T ( r_t^*(a_t^*) - r_t(a_t) )
 *
 * Jika sebuah kebijakan mengalami penghentian awal pada putaran tau < T akibat kebangkrutan
 * anggaran modal gas (B_tau <= B_min), maka pada sisa putaran t in {tau + 1, ..., T}
 * kebijakan tersebut memperoleh imbalan r_t(a_t) = 0 dan menanggung penalti regret penuh r_t^*.
 */
public final class DynamicRegretEvaluator {

    public static final class DynamicRegretStep {
        private final int roundIdx;
        private final int chosenArm;
        private final int oracleArm;
        private final double chosenReward;
        private final double oracleReward;
        private final double chosenCost;
        private final double instantaneousRegret;
        private final double cumulativeRegret;
        private final boolean stateSwitchOccurred;

        public DynamicRegretStep(
                int roundIdx,
                int chosenArm,
                int oracleArm,
                double chosenReward,
                double oracleReward,
                double chosenCost,
                double instantaneousRegret,
                double cumulativeRegret,
                boolean stateSwitchOccurred
        ) {
            this.roundIdx = roundIdx;
            this.chosenArm = chosenArm;
            this.oracleArm = oracleArm;
            this.chosenReward = chosenReward;
            this.oracleReward = oracleReward;
            this.chosenCost = chosenCost;
            this.instantaneousRegret = instantaneousRegret;
            this.cumulativeRegret = cumulativeRegret;
            this.stateSwitchOccurred = stateSwitchOccurred;
        }

        public int getRoundIdx() {
            return roundIdx;
        }

        public double getInstantaneousRegret() {
            return instantaneousRegret;
        }

        public double getCumulativeRegret() {
            return cumulativeRegret;
        }
    }

    public static final class DynamicRegretSummary {
        private final String policyName;
        private final int totalRounds;
        private final int stateSwitches;
        private final double cumulativeDynamicRegret;
        private final double meanDynamicRegret;
        private final double cumulativeReward;
        private final double cumulativeOracleReward;
        private final double theoreticalUpperBound;
        private final List<DynamicRegretStep> trajectory;

        public DynamicRegretSummary(
                String policyName,
                int totalRounds,
                int stateSwitches,
                double cumulativeDynamicRegret,
                double meanDynamicRegret,
                double cumulativeReward,
                double cumulativeOracleReward,
                double theoreticalUpperBound,
                List<DynamicRegretStep> trajectory
        ) {
            this.policyName = policyName;
            this.totalRounds = totalRounds;
            this.stateSwitches = stateSwitches;
            this.cumulativeDynamicRegret = cumulativeDynamicRegret;
            this.meanDynamicRegret = meanDynamicRegret;
            this.cumulativeReward = cumulativeReward;
            this.cumulativeOracleReward = cumulativeOracleReward;
            this.theoreticalUpperBound = theoreticalUpperBound;
            this.trajectory = Collections.unmodifiableList(new ArrayList<>(trajectory));
        }

        public String getPolicyName() {
            return policyName;
        }

        public int getTotalRounds() {
            return totalRounds;
        }

        public int getStateSwitches() {
            return stateSwitches;
        }

        public double getCumulativeDynamicRegret() {
            return cumulativeDynamicRegret;
        }

        public double getMeanDynamicRegret() {
            return meanDynamicRegret;
        }

        public double getCumulativeReward() {
            return cumulativeReward;
        }

        public double getCumulativeOracleReward() {
            return cumulativeOracleReward;
        }

        public double getTheoreticalUpperBound() {
            return theoreticalUpperBound;
        }

        public List<DynamicRegretStep> getTrajectory() {
            return trajectory;
        }
    }

    private final int contextDim;
    private final int nArms;
    private final double initialBudget;
    private final double oracleRewardCeiling;

    public DynamicRegretEvaluator(
            int contextDim,
            int nArms,
            double initialBudget,
            double oracleRewardCeiling
    ) {
        this.contextDim = contextDim;
        this.nArms = nArms;
        this.initialBudget = initialBudget;
        this.oracleRewardCeiling = oracleRewardCeiling;
    }

    public double computeTheoreticalUpperBound(int horizonT, int stateSwitchesS) {
        int t = Math.max(1, horizonT);
        int s = Math.max(0, stateSwitchesS);
        double d = Math.max(1, contextDim);
        double k = Math.max(2, nArms);
        double b = Math.max(1e-3, initialBudget);

        double baseTerm = d * Math.sqrt((1.0 + s) * (double) t * Math.log(1.0 + (double) t));
        double budgetPenalty = (1.0 + ((double) t / (b * 100.0))) * Math.sqrt(Math.log(k) / (double) t);
        return baseTerm * (1.0 + 0.1 * budgetPenalty);
    }

    public DynamicRegretSummary evaluateTrajectory(
            String policyName,
            List<Map<String, Object>> roundLogs,
            Map<Integer, Double> pooledBestRewardPerRound,
            Integer horizonT
    ) {
        List<DynamicRegretStep> steps = new ArrayList<>();
        double cumRegret = 0.0;
        double cumReward = 0.0;
        double cumOracle = 0.0;
        int switches = 0;
        String prevEvent = null;

        int survivedRounds = roundLogs != null ? roundLogs.size() : 0;
        int totalT = horizonT != null ? Math.max(horizonT, survivedRounds) : survivedRounds;

        for (int idx = 0; idx < survivedRounds; idx++) {
            Map<String, Object> log = roundLogs.get(idx);
            int t = log.containsKey("round") ? ((Number) log.get("round")).intValue() : (idx + 1);
            int chosenArm = log.containsKey("arm_id") ? ((Number) log.get("arm_id")).intValue() : 0;
            double rT = log.containsKey("reward") ? ((Number) log.get("reward")).doubleValue() : 0.0;
            double cT = log.containsKey("cost_eth") ? ((Number) log.get("cost_eth")).doubleValue() : 0.001;

            double rStar = pooledBestRewardPerRound != null && pooledBestRewardPerRound.containsKey(t)
                    ? Math.max(rT, pooledBestRewardPerRound.get(t))
                    : Math.max(rT, oracleRewardCeiling);
            rStar = LinearAlgebra.clip(rStar, 0.0, 1.0);

            String evType = String.valueOf(log.getOrDefault("event_type", "none"));
            boolean isSwitch = !"none".equals(evType) && (prevEvent == null || !evType.equals(prevEvent));
            if (isSwitch) {
                switches++;
            }
            prevEvent = evType;

            double instRegret = Math.max(0.0, rStar - rT);
            cumRegret += instRegret;
            cumReward += rT;
            cumOracle += rStar;

            steps.add(new DynamicRegretStep(
                    t, chosenArm, chosenArm, rT, rStar, cT, instRegret, cumRegret, isSwitch
            ));
        }

        // Penalti penghentian awal akibat kebangkrutan anggaran pada t in {tau + 1, ..., T}
        for (int t = survivedRounds + 1; t <= totalT; t++) {
            double rStar = pooledBestRewardPerRound != null && pooledBestRewardPerRound.containsKey(t)
                    ? pooledBestRewardPerRound.get(t)
                    : oracleRewardCeiling;
            rStar = LinearAlgebra.clip(rStar, 0.0, 1.0);
            cumRegret += rStar;
            cumOracle += rStar;
            steps.add(new DynamicRegretStep(t, -1, -1, 0.0, rStar, 0.0, rStar, cumRegret, false));
        }

        int denomT = Math.max(1, totalT);
        double meanRegret = cumRegret / denomT;
        double bound = computeTheoreticalUpperBound(denomT, switches);

        return new DynamicRegretSummary(
                policyName,
                totalT,
                switches,
                cumRegret,
                meanRegret,
                cumReward,
                cumOracle,
                bound,
                steps
        );
    }

    public static Map<String, DynamicRegretSummary> comparePoliciesDynamicRegret(
            Map<String, List<Map<String, Object>>> policiesRoundLogs,
            int contextDim,
            int nArms,
            double initialBudget,
            double oracleRewardCeiling,
            Integer horizonT
    ) {
        Map<Integer, Double> pooledBest = new LinkedHashMap<>();
        int maxObservedRound = 0;

        for (List<Map<String, Object>> logs : policiesRoundLogs.values()) {
            for (int idx = 0; idx < logs.size(); idx++) {
                Map<String, Object> log = logs.get(idx);
                int t = log.containsKey("round") ? ((Number) log.get("round")).intValue() : (idx + 1);
                double r = log.containsKey("reward") ? ((Number) log.get("reward")).doubleValue() : 0.0;
                maxObservedRound = Math.max(maxObservedRound, t);
                double prevBest = pooledBest.getOrDefault(t, 0.20);
                pooledBest.put(t, Math.max(prevBest, Math.min(oracleRewardCeiling, r + 0.15)));
            }
        }

        int targetHorizon = horizonT != null ? Math.max(horizonT, maxObservedRound) : maxObservedRound;
        for (int t = 1; t <= targetHorizon; t++) {
            pooledBest.putIfAbsent(t, Math.min(oracleRewardCeiling, 0.70));
        }

        DynamicRegretEvaluator evaluator = new DynamicRegretEvaluator(
                contextDim, nArms, initialBudget, oracleRewardCeiling
        );
        Map<String, DynamicRegretSummary> summaries = new LinkedHashMap<>();
        for (Map.Entry<String, List<Map<String, Object>>> entry : policiesRoundLogs.entrySet()) {
            summaries.put(
                    entry.getKey(),
                    evaluator.evaluateTrajectory(entry.getKey(), entry.getValue(), pooledBest, targetHorizon)
            );
        }
        return summaries;
    }
}
