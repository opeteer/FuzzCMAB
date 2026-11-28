package id.ac.skripsi.fuzzcmab.vaultcustody;

import id.ac.skripsi.fuzzcmab.algorithms.ArmSelectionResult;
import id.ac.skripsi.fuzzcmab.algorithms.BaseCMABPolicy;
import id.ac.skripsi.fuzzcmab.algorithms.DiscountedLinCBwK;
import id.ac.skripsi.fuzzcmab.algorithms.DiscountedLinUCB;
import id.ac.skripsi.fuzzcmab.algorithms.DisjointLinUCB;
import id.ac.skripsi.fuzzcmab.algorithms.StationaryLinCBwK;
import id.ac.skripsi.fuzzcmab.algorithms.UniformRandomPolicy;
import id.ac.skripsi.fuzzcmab.common.StepTransition;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Orkestrator Agen CMAB untuk Agen_Vault_Custody yang mendukung ke-5 kebijakan:
 *   - d-lincbwk / d-linucbwk (DiscountedLinCBwK)
 *   - lincbwk / linucbwk (StationaryLinCBwK)
 *   - d-linucb (DiscountedLinUCB)
 *   - linucb (DisjointLinUCB)
 *   - uniform / random (UniformRandomPolicy)
 */
public class VaultCustodyCMABAgent {

    public static final class ActionTransitionPair {
        private final ArmSelectionResult selection;
        private final StepTransition transition;

        public ActionTransitionPair(ArmSelectionResult selection, StepTransition transition) {
            this.selection = selection;
            this.transition = transition;
        }

        public ArmSelectionResult getSelection() {
            return selection;
        }

        public StepTransition getTransition() {
            return transition;
        }
    }

    private final VaultCustodyEnv env;
    private final BaseCMABPolicy policy;
    private final List<Map<String, Object>> discoveredVulnerabilities;

    public VaultCustodyCMABAgent(
            VaultCustodyEnv env,
            String algorithm,
            double alpha,
            double l2Lambda,
            double gamma,
            double initialBudget,
            int totalHorizon,
            boolean useTheoreticalBeta,
            boolean decayAllArms,
            long seed
    ) {
        this.env = env;
        this.discoveredVulnerabilities = new ArrayList<>();
        String algoKey = algorithm != null ? algorithm.toLowerCase().trim() : "d-lincbwk";

        if (algoKey.equals("d-lincbwk") || algoKey.equals("d_lincbwk")
                || algoKey.equals("d-linucbwk") || algoKey.equals("d_linucbwk")) {
            this.policy = DiscountedLinCBwK.createDefault(
                    env.getNArms(),
                    PublicContextExtractor.CONTEXT_DIM,
                    gamma,
                    initialBudget,
                    totalHorizon,
                    alpha,
                    l2Lambda,
                    1.0,
                    seed,
                    "D-LinCBwK"
            );
        } else if (algoKey.equals("lincbwk") || algoKey.equals("linucbwk")) {
            this.policy = new StationaryLinCBwK(
                    env.getNArms(),
                    PublicContextExtractor.CONTEXT_DIM,
                    initialBudget,
                    totalHorizon,
                    alpha,
                    l2Lambda,
                    1.0,
                    seed,
                    "LinCBwK"
            );
        } else if (algoKey.equals("d-linucb") || algoKey.equals("d_linucb")) {
            this.policy = new DiscountedLinUCB(
                    env.getNArms(),
                    PublicContextExtractor.CONTEXT_DIM,
                    gamma,
                    alpha,
                    l2Lambda,
                    useTheoreticalBeta,
                    decayAllArms,
                    seed
            );
        } else if (algoKey.equals("uniform") || algoKey.equals("random")) {
            this.policy = new UniformRandomPolicy(
                    env.getNArms(),
                    PublicContextExtractor.CONTEXT_DIM,
                    initialBudget,
                    totalHorizon,
                    seed,
                    "UniformRandom"
            );
        } else {
            this.policy = new DisjointLinUCB(
                    env.getNArms(),
                    PublicContextExtractor.CONTEXT_DIM,
                    alpha,
                    l2Lambda,
                    seed
            );
        }
    }

    public VaultCustodyEnv getEnv() {
        return env;
    }

    public BaseCMABPolicy getPolicy() {
        return policy;
    }

    public List<Map<String, Object>> getDiscoveredVulnerabilities() {
        return discoveredVulnerabilities;
    }

    public ActionTransitionPair actAndLearn() {
        env.prepareRound(env.getRoundCounter() + 1);
        OnChainContextSnapshot prevSnap = env.observeSnapshot();
        double[] contextVec = env.observeContextVector(prevSnap);

        ArmSelectionResult selection = policy.selectArm(contextVec);
        StepTransition transition = env.step(selection.getArmId(), contextVec, prevSnap);

        double reward = transition.getRewardBreakdown().getTotalReward();
        policy.update(selection.getArmId(), contextVec, reward, transition.getTxCostEth());

        for (String anomaly : transition.getRewardBreakdown().getDetectedAnomalies()) {
            Map<String, Object> rec = new LinkedHashMap<>();
            rec.put("round", transition.getRoundIdx());
            rec.put("arm_id", selection.getArmId());
            rec.put("arm_name", transition.getArmSpec().getName());
            rec.put("anomaly", anomaly);
            discoveredVulnerabilities.add(rec);
        }

        return new ActionTransitionPair(selection, transition);
    }
}
