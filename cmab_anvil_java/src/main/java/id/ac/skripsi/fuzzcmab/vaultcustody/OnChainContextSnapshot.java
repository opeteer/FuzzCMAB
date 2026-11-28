package id.ac.skripsi.fuzzcmab.vaultcustody;

/**
 * Snapshot state on-chain dan jejak eksekusi EVM yang diamati oleh Agen_Vault_Custody pada putaran t.
 */
public final class OnChainContextSnapshot {

    private final int blockNumber;
    private final double vaultEthBalance;
    private final double vaultTotalDeposited;
    private final double agentVaultBalance;
    private final double agentEoaEthBalance;
    private final double agentTokenBalance;
    private final boolean vaultIsSolvent;
    private final boolean vaultIsLocked;
    private final boolean vaultCodeAlive;
    private final double recentRevertRate;
    private final boolean sstoreExecuted;
    private final boolean dataflowWaypoint;
    private final double minDepositRequirement;

    public OnChainContextSnapshot(
            int blockNumber,
            double vaultEthBalance,
            double vaultTotalDeposited,
            double agentVaultBalance,
            double agentEoaEthBalance,
            double agentTokenBalance,
            boolean vaultIsSolvent,
            boolean vaultIsLocked,
            boolean vaultCodeAlive,
            double recentRevertRate,
            boolean sstoreExecuted,
            boolean dataflowWaypoint,
            double minDepositRequirement
    ) {
        this.blockNumber = blockNumber;
        this.vaultEthBalance = vaultEthBalance;
        this.vaultTotalDeposited = vaultTotalDeposited;
        this.agentVaultBalance = agentVaultBalance;
        this.agentEoaEthBalance = agentEoaEthBalance;
        this.agentTokenBalance = agentTokenBalance;
        this.vaultIsSolvent = vaultIsSolvent;
        this.vaultIsLocked = vaultIsLocked;
        this.vaultCodeAlive = vaultCodeAlive;
        this.recentRevertRate = recentRevertRate;
        this.sstoreExecuted = sstoreExecuted;
        this.dataflowWaypoint = dataflowWaypoint;
        this.minDepositRequirement = minDepositRequirement;
    }

    public int getBlockNumber() {
        return blockNumber;
    }

    public double getVaultEthBalance() {
        return vaultEthBalance;
    }

    public double getVaultTotalDeposited() {
        return vaultTotalDeposited;
    }

    public double getAgentVaultBalance() {
        return agentVaultBalance;
    }

    public double getAgentEoaEthBalance() {
        return agentEoaEthBalance;
    }

    public double getAgentTokenBalance() {
        return agentTokenBalance;
    }

    public boolean isVaultIsSolvent() {
        return vaultIsSolvent;
    }

    public boolean isVaultIsLocked() {
        return vaultIsLocked;
    }

    public boolean isVaultCodeAlive() {
        return vaultCodeAlive;
    }

    public double getRecentRevertRate() {
        return recentRevertRate;
    }

    public boolean isSstoreExecuted() {
        return sstoreExecuted;
    }

    public boolean isDataflowWaypoint() {
        return dataflowWaypoint;
    }

    public double getMinDepositRequirement() {
        return minDepositRequirement;
    }
}
