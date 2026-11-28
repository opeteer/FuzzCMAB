package id.ac.skripsi.fuzzcmab.algorithms;

import id.ac.skripsi.fuzzcmab.util.LinearAlgebra;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Kebijakan 5 (Metode Usulan Utama): Discounted LinUCB with Knapsacks (D-LinCBwK).
 *
 * Mengimplementasikan Persamaan (3.13) hingga (3.21) dan Algoritma 3.1 pada Bab III:
 *   1. Estimator Imbalan Terdiskonto (Persamaan 3.16):
 *      hat{r}_{t,a}(x_t) = clip( x_t^T hat{theta}_{t,a} + alpha * sqrt(x_t^T A_{t,a}^{-1} x_t), 0, 1 )
 *   2. Estimator Biaya Gas Terdiskonto (Persamaan 3.17):
 *      check{c}_{t,a}(x_t) = max( c_min, x_t^T hat{psi}_{t,a} - beta * sqrt(x_t^T A_{t,a}^{-1} x_t) )
 *   3. Aturan Seleksi Lengan Dual Lagrangian (Persamaan 3.18):
 *      a_t = argmax_{a in A} [ hat{r}_{t,a}(x_t) - lambda_t * check{c}_{t,a}(x_t) ]
 *   4. Pembaruan Kovariansi & Respon Terdiskonto (Persamaan 3.19 - 3.21):
 *      A_{t+1, a_t} = gamma * A_{t, a_t} + x_t x_t^T + (1 - gamma) * lambda_reg * I_d
 *      b_{rew, t+1, a_t} = gamma * b_{rew, t, a_t} + r_t * x_t
 *      b_{cost, t+1, a_t} = gamma * b_{cost, t, a_t} + c_t * x_t
 *   5. Pembaruan Variabel Dual Lagrangian Online (Persamaan 3.13):
 *      lambda_{t+1} = clip( lambda_t + eta * (c_t(a_t) - B / T), 0, lambda_max )
 */
public class DiscountedLinCBwK extends BaseCMABPolicy {

    private final double initialBudget;
    private double remainingBudget;
    private final int totalHorizon;
    private final double gamma;
    private final double alpha;
    private final double beta;
    private final double l2Lambda;
    private final double initialLambda;
    private double lambdaT;
    private final double minCost;
    private final double maxCost;
    private final double minBudgetThreshold;
    private final boolean useShermanMorrison;
    private final boolean decayAllArms;
    private final String dualUpdateMode;
    private final double lambdaMin;
    private final double lambdaMax;
    private final double eta;
    private final Random rng;

    private final double[][][] aMat;
    private final double[][][] aInv;
    private final double[][] bReward;
    private final double[][] theta;
    private final double[][] bCost;
    private final double[][] psi;

    private double lastPredictedCost;
    private double cumulativeCost;
    private final List<Double> dualHistory;
    private final List<Double> budgetHistory;
    private final List<Double> costHistory;
    private final List<Double> rewardHistory;

    public DiscountedLinCBwK(
            int nArms,
            int contextDim,
            double initialBudget,
            int totalHorizon,
            double gamma,
            double alpha,
            double beta,
            double l2Lambda,
            double initialLambda,
            double minCost,
            Double maxCost,
            double minBudgetThreshold,
            boolean useShermanMorrison,
            boolean decayAllArms,
            String dualUpdateMode,
            double lambdaMin,
            double lambdaMax,
            Double customEta,
            long seed,
            String name
    ) {
        super(nArms, contextDim, name != null ? name : "D-LinCBwK");
        if (initialBudget <= 0.0) {
            throw new IllegalArgumentException("initialBudget harus bernilai positif.");
        }
        if (totalHorizon <= 0) {
            throw new IllegalArgumentException("totalHorizon harus bernilai positif.");
        }
        if (gamma <= 0.0 || gamma > 1.0) {
            throw new IllegalArgumentException("gamma harus berada pada (0.0, 1.0].");
        }
        if (alpha < 0.0 || beta < 0.0) {
            throw new IllegalArgumentException("alpha dan beta harus non-negatif.");
        }
        if (l2Lambda <= 0.0 || minCost <= 0.0) {
            throw new IllegalArgumentException("l2Lambda dan minCost harus positif.");
        }

        this.initialBudget = initialBudget;
        this.remainingBudget = initialBudget;
        this.totalHorizon = totalHorizon;
        this.gamma = gamma;
        this.alpha = alpha;
        this.beta = beta;
        this.l2Lambda = l2Lambda;
        this.initialLambda = initialLambda;
        this.lambdaT = initialLambda;
        this.minCost = minCost;
        this.maxCost = maxCost != null ? maxCost : Math.max(initialBudget, 10.0);
        this.minBudgetThreshold = minBudgetThreshold;
        this.useShermanMorrison = useShermanMorrison;
        this.decayAllArms = decayAllArms;
        this.dualUpdateMode = dualUpdateMode != null ? dualUpdateMode : "observed";
        this.lambdaMin = initialLambda == 0.0 ? 0.0 : lambdaMin;
        this.lambdaMax = lambdaMax;
        this.eta = customEta != null
                ? customEta
                : Math.sqrt(Math.log(Math.max(2, nArms)) / (double) totalHorizon);
        this.rng = new Random(seed);

        this.aMat = new double[nArms][contextDim][contextDim];
        this.aInv = new double[nArms][contextDim][contextDim];
        this.bReward = new double[nArms][contextDim];
        this.theta = new double[nArms][contextDim];
        this.bCost = new double[nArms][contextDim];
        this.psi = new double[nArms][contextDim];

        this.dualHistory = new ArrayList<>();
        this.budgetHistory = new ArrayList<>();
        this.costHistory = new ArrayList<>();
        this.rewardHistory = new ArrayList<>();

        reset();
    }

    public static DiscountedLinCBwK createDefault(
            int nArms,
            int contextDim,
            double gamma,
            double initialBudget,
            int totalHorizon,
            double alpha,
            double l2Lambda,
            double initialLambda,
            long seed,
            String name
    ) {
        return new DiscountedLinCBwK(
                nArms,
                contextDim,
                initialBudget,
                totalHorizon,
                gamma,
                alpha,
                alpha,
                l2Lambda,
                initialLambda,
                0.001,
                null,
                0.005,
                true,
                false,
                "observed",
                0.0,
                50.0,
                null,
                seed,
                name
        );
    }

    @Override
    public void reset() {
        this.totalRounds = 0;
        Arrays.fill(this.armCounts, 0L);
        this.cumulativeReward = 0.0;
        this.cumulativeCost = 0.0;
        this.remainingBudget = this.initialBudget;
        this.lambdaT = this.initialLambda;
        this.lastPredictedCost = this.minCost;

        this.dualHistory.clear();
        this.dualHistory.add(this.lambdaT);
        this.budgetHistory.clear();
        this.budgetHistory.add(this.remainingBudget);
        this.costHistory.clear();
        this.rewardHistory.clear();

        for (int a = 0; a < nArms; a++) {
            this.aMat[a] = LinearAlgebra.identity(contextDim, l2Lambda);
            this.aInv[a] = LinearAlgebra.identity(contextDim, 1.0 / l2Lambda);
            Arrays.fill(this.bReward[a], 0.0);
            Arrays.fill(this.theta[a], 0.0);
            Arrays.fill(this.bCost[a], 0.0);
            Arrays.fill(this.psi[a], 0.0);
        }
    }

    @Override
    public double getLambdaT() {
        return lambdaT;
    }

    public double getRemainingBudget() {
        return remainingBudget;
    }

    public double getEta() {
        return eta;
    }

    public double[][][] getAMat() {
        return aMat;
    }

    /**
     * Persamaan (3.16): Estimator Imbalan UCB Terdiskonto [meanEst, uncertainty, ucb].
     */
    public double[] estimateReward(int armId, double[] x) {
        double[] xCol = validateContext(x);
        double meanEst = LinearAlgebra.dot(theta[armId], xCol);
        double quad = LinearAlgebra.quadraticForm(aInv[armId], xCol);
        double uncertainty = alpha * Math.sqrt(quad);
        double ucb = LinearAlgebra.clip(meanEst + uncertainty, 0.0, 1.0);
        return new double[]{meanEst, uncertainty, ucb};
    }

    /**
     * Persamaan (3.17): Estimator Biaya Gas LCB Terdiskonto [expectedCost, predictedCost].
     */
    public double[] estimateCost(int armId, double[] x) {
        double[] xCol = validateContext(x);
        double expectedCost = LinearAlgebra.dot(psi[armId], xCol);
        double predictedCost;
        if (beta > 0.0) {
            double quad = LinearAlgebra.quadraticForm(aInv[armId], xCol);
            double lcbBonus = beta * Math.sqrt(quad);
            predictedCost = Math.max(minCost, expectedCost - lcbBonus);
        } else {
            predictedCost = Math.max(minCost, expectedCost);
        }
        return new double[]{expectedCost, predictedCost};
    }

    @Override
    public ArmSelectionResult selectArm(double[] context, List<Integer> availableArms) {
        return selectArm(context, availableArms, false);
    }

    public ArmSelectionResult selectArm(
            double[] context,
            List<Integer> availableArms,
            boolean filterUnaffordable
    ) {
        double[] x = validateContext(context);
        List<Integer> candidates = resolveCandidateArms(availableArms);

        Map<Integer, Double> allScores = new LinkedHashMap<>();
        Map<Integer, Double> expectedRewards = new LinkedHashMap<>();
        Map<Integer, Double> uncertainties = new LinkedHashMap<>();
        Map<Integer, Double> ucbRewards = new LinkedHashMap<>();
        Map<Integer, Double> predictedCosts = new LinkedHashMap<>();
        Map<Integer, Double> lagrangianPenalties = new LinkedHashMap<>();

        for (int a : candidates) {
            double[] rewEst = estimateReward(a, x);
            double[] costEst = estimateCost(a, x);
            double meanR = rewEst[0];
            double uncR = rewEst[1];
            double ucbR = rewEst[2];
            double predC = costEst[1];

            double penalty = lambdaT * predC;
            double lagrangianScore = ucbR - penalty;

            allScores.put(a, lagrangianScore);
            expectedRewards.put(a, meanR);
            uncertainties.put(a, uncR);
            ucbRewards.put(a, ucbR);
            predictedCosts.put(a, predC);
            lagrangianPenalties.put(a, penalty);
        }

        if (filterUnaffordable && remainingBudget > 0.0) {
            List<Integer> affordable = new ArrayList<>();
            for (int a : candidates) {
                if (predictedCosts.get(a) <= remainingBudget) {
                    affordable.add(a);
                }
            }
            if (!affordable.isEmpty()) {
                candidates = affordable;
            }
        }

        double bestScore = Double.NEGATIVE_INFINITY;
        List<Integer> bestArms = new ArrayList<>();
        for (int a : candidates) {
            double score = allScores.get(a);
            if (score > bestScore + 1e-12) {
                bestScore = score;
                bestArms.clear();
                bestArms.add(a);
            } else if (Math.abs(score - bestScore) <= 1e-12) {
                bestArms.add(a);
            }
        }

        int chosenArm = bestArms.isEmpty() ? candidates.get(0) : LinearAlgebra.chooseRandom(bestArms, rng);
        this.lastPredictedCost = predictedCosts.get(chosenArm);

        int roundsRemaining = Math.max(1, totalHorizon - totalRounds);
        double quotaPerRound = remainingBudget / (double) roundsRemaining;

        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("algorithm", name);
        meta.put("gamma", gamma);
        meta.put("lambda_dual", lambdaT);
        meta.put("remaining_budget", remainingBudget);
        meta.put("quota_per_round", quotaPerRound);
        meta.put("chosen_predicted_cost", predictedCosts.get(chosenArm));
        meta.put("chosen_ucb_reward", ucbRewards.get(chosenArm));
        meta.put("chosen_lagrangian_penalty", lagrangianPenalties.get(chosenArm));

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
        double r = LinearAlgebra.clip(reward, 0.0, 1.0);
        double c = cost != null
                ? LinearAlgebra.clip(cost, minCost, maxCost)
                : estimateCost(armId, x)[1];

        // 1. Pembaruan Matriks Kovariansi & Vektor Respon Terdiskonto (Persamaan 3.19 - 3.21)
        if (gamma < 1.0 - 1e-12) {
            if (decayAllArms) {
                for (int a = 0; a < nArms; a++) {
                    if (a == armId) {
                        LinearAlgebra.updateDiscountedCovarianceInPlace(aMat[a], x, gamma, l2Lambda);
                        LinearAlgebra.updateDiscountedResponseInPlace(bReward[a], x, gamma, r);
                        LinearAlgebra.updateDiscountedResponseInPlace(bCost[a], x, gamma, c);
                    } else {
                        LinearAlgebra.decayCovarianceInPlace(aMat[a], gamma, l2Lambda);
                        for (int i = 0; i < contextDim; i++) {
                            bReward[a][i] *= gamma;
                            bCost[a][i] *= gamma;
                        }
                    }
                    aInv[a] = LinearAlgebra.invert(aMat[a]);
                    theta[a] = LinearAlgebra.matVec(aInv[a], bReward[a]);
                    psi[a] = LinearAlgebra.matVec(aInv[a], bCost[a]);
                }
            } else {
                // Algoritma 3.1 baris 27-32: Hanya lengan terpilih a_t yang diperbarui dan didiskon
                LinearAlgebra.updateDiscountedCovarianceInPlace(aMat[armId], x, gamma, l2Lambda);
                LinearAlgebra.updateDiscountedResponseInPlace(bReward[armId], x, gamma, r);
                LinearAlgebra.updateDiscountedResponseInPlace(bCost[armId], x, gamma, c);

                aInv[armId] = LinearAlgebra.invert(aMat[armId]);
                theta[armId] = LinearAlgebra.matVec(aInv[armId], bReward[armId]);
                psi[armId] = LinearAlgebra.matVec(aInv[armId], bCost[armId]);
            }
        } else {
            // Kasus stasioner (gamma = 1.0)
            for (int i = 0; i < contextDim; i++) {
                for (int j = 0; j < contextDim; j++) {
                    aMat[armId][i][j] += x[i] * x[j];
                }
                bReward[armId][i] += r * x[i];
                bCost[armId][i] += c * x[i];
            }
            if (useShermanMorrison) {
                LinearAlgebra.shermanMorrisonUpdateInPlace(aInv[armId], x);
            } else {
                aInv[armId] = LinearAlgebra.invert(aMat[armId]);
            }
            theta[armId] = LinearAlgebra.matVec(aInv[armId], bReward[armId]);
            psi[armId] = LinearAlgebra.matVec(aInv[armId], bCost[armId]);
        }

        // 2. Pembaruan Anggaran & Variabel Dual Lagrangian (Persamaan 3.12 & 3.13)
        totalRounds++;
        armCounts[armId]++;
        cumulativeReward += r;
        cumulativeCost += c;
        remainingBudget = Math.max(0.0, remainingBudget - c);

        double targetRate = initialBudget / (double) Math.max(1, totalHorizon);
        double dualCostSignal = "predicted".equalsIgnoreCase(dualUpdateMode) ? lastPredictedCost : c;
        double dualGradient = dualCostSignal - targetRate;
        lambdaT = LinearAlgebra.clip(lambdaT + eta * dualGradient, lambdaMin, lambdaMax);

        dualHistory.add(lambdaT);
        budgetHistory.add(remainingBudget);
        costHistory.add(c);
        rewardHistory.add(r);
    }

    public boolean isStopConditionMet() {
        return totalRounds >= totalHorizon || remainingBudget <= minBudgetThreshold;
    }

    @Override
    public Map<String, Object> getDiagnostics() {
        Map<String, Object> diag = new LinkedHashMap<>();
        diag.put("policy_name", name);
        diag.put("gamma", gamma);
        diag.put("total_rounds", totalRounds);
        diag.put("cumulative_reward", cumulativeReward);
        diag.put("mean_reward", totalRounds > 0 ? cumulativeReward / totalRounds : 0.0);
        diag.put("cumulative_cost", cumulativeCost);
        diag.put("initial_budget", initialBudget);
        diag.put("remaining_budget", remainingBudget);
        diag.put("lambda_dual", lambdaT);
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
