package id.ac.skripsi.fuzzcmab.common;

import id.ac.skripsi.fuzzcmab.vaultcustody.BanditRewardBreakdown;
import id.ac.skripsi.fuzzcmab.vaultcustody.OnChainContextSnapshot;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.Map;

/**
 * Kelas Dasar Abstrak untuk lingkungan CMAB agen pengguna biasa (regular user agent) di Anvil.
 */
public abstract class BaseRegularUserBanditEnv {

    protected final RegularUserAnvilClient rpcClient;
    protected final List<Arm> arms;
    protected final int revertWindow;
    protected final Deque<Integer> recentReverts;
    protected int roundCounter;

    protected BaseRegularUserBanditEnv(
            RegularUserAnvilClient rpcClient,
            List<Arm> arms,
            int revertWindow
    ) {
        this.rpcClient = rpcClient;
        this.arms = Collections.unmodifiableList(arms);
        this.revertWindow = Math.max(1, revertWindow);
        this.recentReverts = new ArrayDeque<>(this.revertWindow);
        this.roundCounter = 0;
    }

    public RegularUserAnvilClient getRpcClient() {
        return rpcClient;
    }

    public List<Arm> getArms() {
        return arms;
    }

    public int getNArms() {
        return arms.size();
    }

    public int getRoundCounter() {
        return roundCounter;
    }

    public double getRecentRevertRate() {
        if (recentReverts.isEmpty()) {
            return 0.0;
        }
        int sum = 0;
        for (int v : recentReverts) {
            sum += v;
        }
        return (double) sum / (double) recentReverts.size();
    }

    protected void recordRevertOutcome(boolean txSucceeded) {
        if (recentReverts.size() >= revertWindow) {
            recentReverts.pollFirst();
        }
        recentReverts.addLast(txSucceeded ? 0 : 1);
    }

    public abstract Map<String, String> verifyRegularUserStanding();

    public abstract OnChainContextSnapshot observeSnapshot();

    public abstract double[] observeContextVector(OnChainContextSnapshot snapshot);

    public abstract RegularUserAnvilClient.TxExecutionResult executeArm(Arm arm);

    public abstract BanditRewardBreakdown evaluateReward(
            OnChainContextSnapshot prevSnap,
            OnChainContextSnapshot nextSnap,
            boolean txSucceeded,
            int armId
    );

    public abstract StepTransition step(int armId);
}
