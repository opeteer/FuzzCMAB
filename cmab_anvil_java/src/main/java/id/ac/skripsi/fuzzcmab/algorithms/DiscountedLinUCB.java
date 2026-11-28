package id.ac.skripsi.fuzzcmab.algorithms;

import id.ac.skripsi.fuzzcmab.util.LinearAlgebra;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Kebijakan 3: Discounted LinUCB / D-LinUCB (Non-Stasioner Tanpa Knapsack: gamma in (0, 1), lambda_t = 0).
 *
 * Reduksi ablasi dari D-LinCBwK (Sub-bab 3.3.5) dengan menetapkan lambda_t = 0 pada Persamaan (3.18):
 *   A_{t+1, a_t} = gamma * A_{t, a_t} + x_t * x_t^T + (1 - gamma) * lambda_reg * I_d
 *   b_{t+1, a_t} = gamma * b_{t, a_t} + r_t * x_t
 *   hat{theta}_{t+1, a_t} = A_{t+1, a_t}^{-1} b_{t+1, a_t}
 */
public class DiscountedLinUCB extends BaseCMABPolicy {

    private final double gamma;
    private final double alpha;
    private final double l2Lambda;
    private final boolean useTheoreticalBeta;
    private final boolean decayAllArms;
    private final Random rng;

    private final double[][][] vMat;
    private final double[][][] vInv;
    private final double[][][] vTilde;
    private final double[][] bVec;
    private final double[][] theta;
    private final double[] effectivePulls;

    public DiscountedLinUCB(
            int nArms,
            int contextDim,
            double gamma,
            double alpha,
            double l2Lambda,
            boolean useTheoreticalBeta,
            boolean decayAllArms,
            long seed
    ) {
        super(nArms, contextDim, String.format("DiscountedLinUCB(gamma=%.2f)", gamma));
        if (gamma <= 0.0 || gamma > 1.0) {
            throw new IllegalArgumentException("gamma harus berada pada (0.0, 1.0].");
        }
        if (alpha < 0.0) {
            throw new IllegalArgumentException("alpha harus non-negatif.");
        }
        if (l2Lambda <= 0.0) {
            throw new IllegalArgumentException("l2Lambda harus positif.");
        }
        this.gamma = gamma;
        this.alpha = alpha;
        this.l2Lambda = l2Lambda;
        this.useTheoreticalBeta = useTheoreticalBeta;
        this.decayAllArms = decayAllArms;
        this.rng = new Random(seed);

        this.vMat = new double[nArms][contextDim][contextDim];
        this.vInv = new double[nArms][contextDim][contextDim];
        this.vTilde = new double[nArms][contextDim][contextDim];
        this.bVec = new double[nArms][contextDim];
        this.theta = new double[nArms][contextDim];
        this.effectivePulls = new double[nArms];
        reset();
    }

    @Override
    public void reset() {
        this.totalRounds = 0;
        Arrays.fill(this.armCounts, 0L);
        this.cumulativeReward = 0.0;
        Arrays.fill(this.effectivePulls, 0.0);

        for (int a = 0; a < nArms; a++) {
            this.vMat[a] = LinearAlgebra.identity(contextDim, l2Lambda);
            this.vInv[a] = LinearAlgebra.identity(contextDim, 1.0 / l2Lambda);
            this.vTilde[a] = LinearAlgebra.identity(contextDim, l2Lambda);
            Arrays.fill(this.bVec[a], 0.0);
            Arrays.fill(this.theta[a], 0.0);
        }
    }

    private double computeBeta() {
        if (!useTheoreticalBeta) {
            return alpha;
        }
        double effHorizon = gamma < 1.0 - 1e-12
                ? (1.0 - Math.pow(gamma, 2.0 * Math.max(totalRounds, 1))) / (1.0 - gamma * gamma)
                : Math.max(totalRounds, 1);
        double d = contextDim;
        double logDet = d * Math.log(1.0 + effHorizon / (l2Lambda * d));
        double confTerm = 0.5 * Math.sqrt(2.0 * Math.log(1.0 / 0.05) + logDet);
        double biasTerm = Math.sqrt(l2Lambda);
        return alpha * (biasTerm + confTerm);
    }

    private double computeConfidenceRadius(int armId, double[] x) {
        if (!useTheoreticalBeta) {
            return Math.sqrt(LinearAlgebra.quadraticForm(vInv[armId], x));
        }
        return Math.sqrt(LinearAlgebra.sandwichQuadraticForm(vInv[armId], vTilde[armId], x));
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
        double betaVal = computeBeta();

        for (int a : candidates) {
            double meanEst = LinearAlgebra.dot(theta[a], x);
            double radius = computeConfidenceRadius(a, x);
            double bonus = betaVal * radius;
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
        meta.put("gamma", gamma);
        meta.put("beta_chosen", betaVal);
        meta.put("effective_pulls_chosen", effectivePulls[chosenArm]);

        return new ArmSelectionResult(
                chosenArm,
                allScores.get(chosenArm),
                expectedRewards.get(chosenArm),
                uncertainties.get(chosenArm),
                allScores,
                meta
        );
    }

    private void decayArm(int armId) {
        if (gamma >= 1.0 - 1e-12) {
            return;
        }
        LinearAlgebra.decayCovarianceInPlace(vMat[armId], gamma, l2Lambda);
        LinearAlgebra.decayCovarianceInPlace(vTilde[armId], gamma * gamma, l2Lambda);
        for (int i = 0; i < contextDim; i++) {
            bVec[armId][i] *= gamma;
        }
        effectivePulls[armId] *= gamma;
    }

    @Override
    public void update(int armId, double[] context, double reward, Double cost) {
        if (armId < 0 || armId >= nArms) {
            throw new IllegalArgumentException("armId tidak valid: " + armId);
        }
        double[] x = validateContext(context);
        double r = reward;

        if (decayAllArms) {
            for (int a = 0; a < nArms; a++) {
                decayArm(a);
            }
        } else {
            decayArm(armId);
        }

        for (int i = 0; i < contextDim; i++) {
            for (int j = 0; j < contextDim; j++) {
                double xx = x[i] * x[j];
                vMat[armId][i][j] += xx;
                vTilde[armId][i][j] += xx;
            }
            bVec[armId][i] += r * x[i];
        }
        effectivePulls[armId] += 1.0;

        if (decayAllArms && gamma < 1.0 - 1e-12) {
            for (int a = 0; a < nArms; a++) {
                vInv[a] = LinearAlgebra.invert(vMat[a]);
                theta[a] = LinearAlgebra.matVec(vInv[a], bVec[a]);
            }
        } else {
            vInv[armId] = LinearAlgebra.invert(vMat[armId]);
            theta[armId] = LinearAlgebra.matVec(vInv[armId], bVec[armId]);
        }

        totalRounds++;
        armCounts[armId]++;
        cumulativeReward += r;
    }

    @Override
    public Map<String, Object> getDiagnostics() {
        Map<String, Object> diag = new LinkedHashMap<>();
        diag.put("policy_name", name);
        diag.put("gamma", gamma);
        diag.put("total_rounds", totalRounds);
        diag.put("cumulative_reward", cumulativeReward);
        diag.put("mean_reward", totalRounds > 0 ? cumulativeReward / totalRounds : 0.0);
        diag.put("arm_counts", Arrays.copyOf(armCounts, nArms));
        diag.put("effective_pulls", Arrays.copyOf(effectivePulls, nArms));
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
