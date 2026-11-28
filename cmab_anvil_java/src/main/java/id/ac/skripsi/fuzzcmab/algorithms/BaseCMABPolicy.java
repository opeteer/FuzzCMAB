package id.ac.skripsi.fuzzcmab.algorithms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Kelas Dasar Abstrak (Abstract Base Class) untuk strategi Contextual Multi-Armed Bandit (CMAB).
 *
 * Pada setiap putaran diskrit t = 1, ..., T:
 *   1. Agen mengamati vektor konteks x_t in R^d.
 *   2. Agen memilih lengan a_t in A.
 *   3. Lingkungan memberikan umpan balik bandit (r_t, c_t).
 *   4. Agen memperbarui parameter internalnya untuk meminimalkan Cumulative Dynamic Regret.
 */
public abstract class BaseCMABPolicy {

    protected final int nArms;
    protected final int contextDim;
    protected final String name;
    protected int totalRounds;
    protected final long[] armCounts;
    protected double cumulativeReward;

    protected BaseCMABPolicy(int nArms, int contextDim, String name) {
        if (nArms <= 0) {
            throw new IllegalArgumentException("nArms harus bernilai positif.");
        }
        if (contextDim <= 0) {
            throw new IllegalArgumentException("contextDim harus bernilai positif.");
        }
        this.nArms = nArms;
        this.contextDim = contextDim;
        this.name = name != null ? name : "BaseCMAB";
        this.totalRounds = 0;
        this.armCounts = new long[nArms];
        this.cumulativeReward = 0.0;
    }

    public int getNArms() {
        return nArms;
    }

    public int getContextDim() {
        return contextDim;
    }

    public String getName() {
        return name;
    }

    public int getTotalRounds() {
        return totalRounds;
    }

    public long[] getArmCounts() {
        return Arrays.copyOf(armCounts, armCounts.length);
    }

    public double getCumulativeReward() {
        return cumulativeReward;
    }

    public double getLambdaT() {
        return 0.0;
    }

    public double[] validateContext(double[] context) {
        if (context == null || context.length != contextDim) {
            int got = context == null ? 0 : context.length;
            throw new IllegalArgumentException(
                    "Ketidakcocokan dimensi konteks: diharapkan " + contextDim + ", diterima " + got
            );
        }
        return Arrays.copyOf(context, contextDim);
    }

    protected List<Integer> resolveCandidateArms(List<Integer> availableArms) {
        if (availableArms == null || availableArms.isEmpty()) {
            List<Integer> all = new ArrayList<>(nArms);
            for (int a = 0; a < nArms; a++) {
                all.add(a);
            }
            return all;
        }
        return new ArrayList<>(availableArms);
    }

    public ArmSelectionResult selectArm(double[] context) {
        return selectArm(context, null);
    }

    public abstract ArmSelectionResult selectArm(double[] context, List<Integer> availableArms);

    public void update(int armId, double[] context, double reward) {
        update(armId, context, reward, null);
    }

    public abstract void update(int armId, double[] context, double reward, Double cost);

    public abstract void reset();

    public abstract Map<String, Object> getDiagnostics();

    public abstract double[][] getTheta();
}
