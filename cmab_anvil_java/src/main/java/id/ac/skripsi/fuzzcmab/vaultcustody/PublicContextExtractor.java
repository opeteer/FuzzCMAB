package id.ac.skripsi.fuzzcmab.vaultcustody;

import id.ac.skripsi.fuzzcmab.util.LinearAlgebra;

import java.util.List;

/**
 * Pemodelan Ruang Konteks (X_Vault subset R^10) untuk Agen_Vault_Custody (Tabel 3.1 Bab III).
 *
 * Memetakan OnChainContextSnapshot menjadi vektor konteks 10 dimensi x_t in R^10 yang
 * bebas-skala (scale-invariant) dan menggabungkan rasio finansial on-chain dengan
 * sinyal jejak eksekusi opcode EVM (SSTORE & SLOAD->SSTORE dataflow waypoint).
 */
public class PublicContextExtractor {

    public static final int CONTEXT_DIM = 10;
    public static final double EPS = 1e-5;

    public static final List<String> FEATURE_NAMES = List.of(
            "x0_bias",
            "x1_vault_liquidity_ratio",
            "x2_claim_liability_ratio",
            "x3_solvency_gap_ratio",
            "x4_agent_stake_ratio",
            "x5_deposit_eligibility",
            "x6_sstore_execution_flag",
            "x7_dataflow_waypoint_flag",
            "x8_recent_revert_rate",
            "x9_vault_code_alive"
    );

    private final double minDepositEth;

    public PublicContextExtractor(double minDepositEth) {
        this.minDepositEth = Math.max(minDepositEth, EPS);
    }

    public double getMinDepositEth() {
        return minDepositEth;
    }

    public double[] toVector(OnChainContextSnapshot snap) {
        double vaultEth = Math.max(0.0, snap.getVaultEthBalance());
        double totalDep = Math.max(0.0, snap.getVaultTotalDeposited());
        double agentStake = Math.max(0.0, snap.getAgentVaultBalance());
        double agentEth = Math.max(0.0, snap.getAgentEoaEthBalance());

        // x0: Konstanta intersep regresi linier UCB
        double x0Bias = 1.0;

        // x1: Relative Vault Liquidity Ratio in [0.0, 1.0]
        double x1VaultLiqRatio = LinearAlgebra.clip(
                vaultEth / (vaultEth + agentEth + EPS), 0.0, 1.0
        );

        // x2: Relative Claim Liability Ratio in [0.0, 1.0]
        double x2LiabilityRatio = LinearAlgebra.clip(
                totalDep / (vaultEth + totalDep + EPS), 0.0, 1.0
        );

        // x3: Solvency Gap Ratio in [-1.0, 1.0]
        double denomSolvency = Math.max(totalDep, 1.0);
        double x3SolvencyGap = LinearAlgebra.clip(
                (vaultEth - totalDep) / denomSolvency, -1.0, 1.0
        );

        // x4: Proporsi kepemilikan stake agen di dalam Vault in [0.0, 1.0]
        double stakeDenom = (totalDep >= agentStake && totalDep > 0.0)
                ? (totalDep + EPS)
                : (Math.max(vaultEth, agentStake) + EPS);
        double x4AgentStakeRatio = LinearAlgebra.clip(agentStake / stakeDenom, 0.0, 1.0);

        // x5: Instant Deposit Eligibility Ratio in [0.0, 1.0]
        double reqDep = Math.max(snap.getMinDepositRequirement() > 0.0 ? snap.getMinDepositRequirement() : minDepositEth, EPS);
        double x5DepositEligibility = LinearAlgebra.clip(Math.min(1.0, agentEth / reqDep), 0.0, 1.0);

        // x6: EVM SSTORE Execution Flag in {0.0, 1.0}
        double x6SstoreFlag = snap.isSstoreExecuted() ? 1.0 : 0.0;

        // x7: SMARTIAN / ItyFuzz Dataflow Waypoint Flag in {0.0, 1.0}
        double x7WaypointFlag = snap.isDataflowWaypoint() ? 1.0 : 0.0;

        // x8: Moving-average transaction revert rate in [0.0, 1.0]
        double x8RevertRate = LinearAlgebra.clip(snap.getRecentRevertRate(), 0.0, 1.0);

        // x9: Contract bytecode liveness indicator in {0.0, 1.0}
        double x9CodeAlive = snap.isVaultCodeAlive() ? 1.0 : 0.0;

        return new double[]{
                x0Bias,
                x1VaultLiqRatio,
                x2LiabilityRatio,
                x3SolvencyGap,
                x4AgentStakeRatio,
                x5DepositEligibility,
                x6SstoreFlag,
                x7WaypointFlag,
                x8RevertRate,
                x9CodeAlive
        };
    }
}
