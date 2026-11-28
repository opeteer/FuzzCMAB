package id.ac.skripsi.fuzzcmab.common;

import id.ac.skripsi.fuzzcmab.vaultcustody.BanditRewardBreakdown;
import id.ac.skripsi.fuzzcmab.vaultcustody.OnChainContextSnapshot;

import java.util.Arrays;

/**
 * Catatan transisi lengkap dari satu putaran interaksi CMAB t di simulator Anvil.
 */
public final class StepTransition {

    private final int roundIdx;
    private final Arm armSpec;
    private final boolean txSucceeded;
    private final String txHashOrErr;
    private final OnChainContextSnapshot prevSnapshot;
    private final OnChainContextSnapshot nextSnapshot;
    private final double[] contextVector;
    private final double[] nextContextVector;
    private final BanditRewardBreakdown rewardBreakdown;
    private final double gasPriceGwei;
    private final long gasUsed;
    private final double txCostEth;
    private final double remainingBudgetEth;
    private final String externalEvent;

    public StepTransition(
            int roundIdx,
            Arm armSpec,
            boolean txSucceeded,
            String txHashOrErr,
            OnChainContextSnapshot prevSnapshot,
            OnChainContextSnapshot nextSnapshot,
            double[] contextVector,
            double[] nextContextVector,
            BanditRewardBreakdown rewardBreakdown,
            double gasPriceGwei,
            long gasUsed,
            double txCostEth,
            double remainingBudgetEth,
            String externalEvent
    ) {
        this.roundIdx = roundIdx;
        this.armSpec = armSpec;
        this.txSucceeded = txSucceeded;
        this.txHashOrErr = txHashOrErr;
        this.prevSnapshot = prevSnapshot;
        this.nextSnapshot = nextSnapshot;
        this.contextVector = Arrays.copyOf(contextVector, contextVector.length);
        this.nextContextVector = Arrays.copyOf(nextContextVector, nextContextVector.length);
        this.rewardBreakdown = rewardBreakdown;
        this.gasPriceGwei = gasPriceGwei;
        this.gasUsed = gasUsed;
        this.txCostEth = txCostEth;
        this.remainingBudgetEth = remainingBudgetEth;
        this.externalEvent = externalEvent != null ? externalEvent : "none";
    }

    public int getRoundIdx() {
        return roundIdx;
    }

    public Arm getArmSpec() {
        return armSpec;
    }

    public boolean isTxSucceeded() {
        return txSucceeded;
    }

    public String getTxHashOrErr() {
        return txHashOrErr;
    }

    public OnChainContextSnapshot getPrevSnapshot() {
        return prevSnapshot;
    }

    public OnChainContextSnapshot getNextSnapshot() {
        return nextSnapshot;
    }

    public double[] getContextVector() {
        return Arrays.copyOf(contextVector, contextVector.length);
    }

    public double[] getNextContextVector() {
        return Arrays.copyOf(nextContextVector, nextContextVector.length);
    }

    public BanditRewardBreakdown getRewardBreakdown() {
        return rewardBreakdown;
    }

    public double getGasPriceGwei() {
        return gasPriceGwei;
    }

    public long getGasUsed() {
        return gasUsed;
    }

    public double getTxCostEth() {
        return txCostEth;
    }

    public double getRemainingBudgetEth() {
        return remainingBudgetEth;
    }

    public String getExternalEvent() {
        return externalEvent;
    }
}
