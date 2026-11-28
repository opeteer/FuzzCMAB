package id.ac.skripsi.fuzzcmab.algorithms;

import id.ac.skripsi.fuzzcmab.util.LinearAlgebra;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Kebijakan 2: Disjoint LinUCB (Stasioner Tanpa Kendali Knapsack: gamma = 1.0, lambda_t = 0).
 *
 * Kasus khusus stasioner dari Persamaan (3.16)-(3.21) dengan gamma = 1.0 dan lambda_t = 0:
 *   hat{r}_{t,a}(x_t) = clip( x_t^T hat{theta}_{t,a} + alpha * sqrt(x_t^T A_{t,a}^{-1} x_t), 0, 1 )
 *   a_t = argmax_{a in A} hat{r}_{t,a}(x_t)
 */
public class DisjointLinUCB extends BaseCMABPolicy {

    private final double alpha;
    private final double l2Lambda;
    private final boolean useShermanMorrison;
    private final Random rng;

    private final double[][][] aMat;
    private final double[][][] aInv;
    private final double[][] bVec;
    private final double[][] theta;

    public DisjointLinUCB(
            int nArms,
            int contextDim,
            double alpha,
            double l2Lambda,
            long seed
    ) {
        super(nArms, contextDim, "DisjointLinUCB");
        if (alpha < 0.0) {
            throw new IllegalArgumentException("alpha harus non-negatif.");
        }
        if (l2Lambda <= 0.0) {
            throw new IllegalArgumentException("l2Lambda harus positif.");
        }
        this.alpha = alpha;
        this.l2Lambda = l2Lambda;
        this.useShermanMorrison = true;
        this.rng = new Random(seed);

        this.aMat = new double[nArms][contextDim][contextDim];
        this.aInv = new double[nArms][contextDim][contextDim];
        this.bVec = new double[nArms][contextDim];
        this.theta = new double[nArms][contextDim];
        reset();
    }

    @Override
    public void reset() {
        this.totalRounds = 0;
        Arrays.fill(this.armCounts, 0L);
        this.cumulativeReward = 0.0;

        for (int a = 0; a < nArms; a++) {
            this.aMat[a] = LinearAlgebra.identity(contextDim, l2Lambda);
            this.aInv[a] = LinearAlgebra.identity(contextDim, 1.0 / l2Lambda);
            Arrays.fill(this.bVec[a], 0.0);
            Arrays.fill(this.theta[a], 0.0);
        }
    }

    @Override
    public ArmSelectionResult selectArm(double[] context, List<Integer> availableArms) {
        double[] x = validateContext(context);
        List<Integer> candidates = resolveCandidateArms(availableArms);

        Map<Integer, Double> allScores = new LinkedHashMap<>();
        Map<Integer, Double> expectedRewards = new LinkedHashMap<>();
        Map<Integer, Double> uncertainties = new LinkedHashMap<>();

        double bestScore = Double.NEGATIVE_INFINITY;
        List<Integer> bestArms = new ArrayList<>();

        for (int a : candidates) {
            double meanEst = LinearAlgebra.dot(theta[a], x);
            double quad = LinearAlgebra.quadraticForm(aInv[a], x);
            double bonus = alpha * Math.sqrt(quad);
            double ucb = LinearAlgebra.clip(meanEst + bonus, 0.0, 1.0);

            allScores.put(a, ucb);
            expectedRewards.put(a, meanEst);
            uncertainties.put(a, bonus);

            if (ucb > bestScore + 1e-12) {
                bestScore = ucb;
                bestArms.clear();
                bestArms.add(a);
            } else if (Math.abs(ucb - bestScore) <= 1e-12) {
                bestArms.add(a);
            }
        }

        int chosenArm = LinearAlgebra.chooseRandom(bestArms, rng);
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("alpha_t", alpha);
        meta.put("arm_pulls", armCounts[chosenArm]);

        return new ArmSelectionResult(
                chosenArm,
                allScores.get(chosenArm),
                expectedRewards.get(chosenArm),
                uncertainties.get(chosenArm),
                allScores,
                meta
        );
    }

    @Override
    public void update(int armId, double[] context, double reward, Double cost) {
        if (armId < 0 || armId >= nArms) {
            throw new IllegalArgumentException("armId tidak valid: " + armId);
        }
        double[] x = validateContext(context);
        double r = reward;

        for (int i = 0; i < contextDim; i++) {
            for (int j = 0; j < contextDim; j++) {
                aMat[armId][i][j] += x[i] * x[j];
            }
            bVec[armId][i] += r * x[i];
        }

        if (useShermanMorrison) {
            LinearAlgebra.shermanMorrisonUpdateInPlace(aInv[armId], x);
        } else {
            aInv[armId] = LinearAlgebra.invert(aMat[armId]);
        }

        theta[armId] = LinearAlgebra.matVec(aInv[armId], bVec[armId]);
        totalRounds++;
        armCounts[armId]++;
        cumulativeReward += r;
    }

    @Override
    public Map<String, Object> getDiagnostics() {
        Map<String, Object> diag = new LinkedHashMap<>();
        diag.put("policy_name", name);
        diag.put("total_rounds", totalRounds);
        diag.put("cumulative_reward", cumulativeReward);
        diag.put("mean_reward", totalRounds > 0 ? cumulativeReward / totalRounds : 0.0);
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
