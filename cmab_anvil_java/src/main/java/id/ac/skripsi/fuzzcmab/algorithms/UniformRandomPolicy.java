package id.ac.skripsi.fuzzcmab.algorithms;

import id.ac.skripsi.fuzzcmab.util.LinearAlgebra;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Kebijakan 1: Uniform Random Policy Baseline (Representasi Forge Fuzz / AFL).
 *
 * Memilih lengan secara acak seragam P(a_t = a) = 1 / |Candidates| tanpa regresi konteks
 * maupun pengali dual Knapsack (lambda_t = 0).
 */
public class UniformRandomPolicy extends BaseCMABPolicy {

    private final double initialBudget;
    private double remainingBudget;
    private final int totalHorizon;
    private final Random rng;

    private double cumulativeCost;
    private final double[] empiricalSumReward;
    private final double[] empiricalSumCost;
    private final double[][] theta;

    public UniformRandomPolicy(
            int nArms,
            int contextDim,
            double initialBudget,
            int totalHorizon,
            long seed,
            String name
    ) {
        super(nArms, contextDim, name != null ? name : "UniformRandom");
        this.initialBudget = initialBudget;
        this.remainingBudget = initialBudget;
        this.totalHorizon = totalHorizon;
        this.rng = new Random(seed);
        this.empiricalSumReward = new double[nArms];
        this.empiricalSumCost = new double[nArms];
        this.theta = new double[nArms][contextDim];
        reset();
    }

    @Override
    public void reset() {
        this.totalRounds = 0;
        Arrays.fill(this.armCounts, 0L);
        this.cumulativeReward = 0.0;
        this.cumulativeCost = 0.0;
        this.remainingBudget = this.initialBudget;
        Arrays.fill(this.empiricalSumReward, 0.0);
        Arrays.fill(this.empiricalSumCost, 0.0);
        for (int a = 0; a < nArms; a++) {
            Arrays.fill(this.theta[a], 0.0);
        }
    }

    @Override
    public ArmSelectionResult selectArm(double[] context, List<Integer> availableArms) {
        validateContext(context);
        List<Integer> candidates = resolveCandidateArms(availableArms);
        int chosenArm = LinearAlgebra.chooseRandom(candidates, rng);
        double prob = 1.0 / Math.max(1, candidates.size());

        Map<Integer, Double> allScores = new LinkedHashMap<>();
        for (int a : candidates) {
            allScores.put(a, prob);
        }

        long pulls = armCounts[chosenArm];
        double empMean = pulls > 0 ? (empiricalSumReward[chosenArm] / pulls) : 0.0;

        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("algorithm", name);
        meta.put("strategy", "uniform_random");
        meta.put("prob", prob);
        meta.put("remaining_budget", remainingBudget);

        return new ArmSelectionResult(chosenArm, prob, empMean, 0.0, allScores, meta);
    }

    @Override
    public void update(int armId, double[] context, double reward, Double cost) {
        if (armId < 0 || armId >= nArms) {
            throw new IllegalArgumentException("armId tidak valid: " + armId);
        }
        validateContext(context);
        double r = LinearAlgebra.clip(reward, 0.0, 1.0);
        double c = cost != null ? Math.max(0.0, cost) : 0.0;

        totalRounds++;
        armCounts[armId]++;
        cumulativeReward += r;
        cumulativeCost += c;
        remainingBudget = Math.max(0.0, remainingBudget - c);

        empiricalSumReward[armId] += r;
        empiricalSumCost[armId] += c;
        theta[armId][0] = empiricalSumReward[armId] / armCounts[armId];
    }

    @Override
    public Map<String, Object> getDiagnostics() {
        Map<String, Object> diag = new LinkedHashMap<>();
        diag.put("policy_name", name);
        diag.put("total_rounds", totalRounds);
        diag.put("cumulative_reward", cumulativeReward);
        diag.put("mean_reward", totalRounds > 0 ? cumulativeReward / totalRounds : 0.0);
        diag.put("cumulative_cost", cumulativeCost);
        diag.put("remaining_budget", remainingBudget);
        diag.put("arm_counts", Arrays.copyOf(armCounts, nArms));
        List<Double> norms = new ArrayList<>(nArms);
        for (int a = 0; a < nArms; a++) {
            norms.add(LinearAlgebra.l2Norm(theta[a]));
        }
        diag.put("theta_l2_norms", norms);
        return diag;
    }

    @Override
    public double[][] getTheta() {
        return LinearAlgebra.copyMatrix(theta);
    }
}
