package id.ac.skripsi.fuzzcmab.common;

/**
 * Spesifikasi lengan bandit diskrit (Immutable Value Object) yang merepresentasikan
 * satu strategi transaksi EOA agen pengguna biasa.
 */
public final class Arm {

    private final int armId;
    private final String name;
    private final String targetContract;
    private final String functionSig;
    private final double ethValue;
    private final String description;

    public Arm(
            int armId,
            String name,
            String targetContract,
            String functionSig,
            double ethValue,
            String description
    ) {
        this.armId = armId;
        this.name = name;
        this.targetContract = targetContract;
        this.functionSig = functionSig != null ? functionSig : "";
        this.ethValue = ethValue;
        this.description = description != null ? description : "";
    }

    public int getArmId() {
        return armId;
    }

    public String getName() {
        return name;
    }

    public String getTargetContract() {
        return targetContract;
    }

    public String getFunctionSig() {
        return functionSig;
    }

    public double getEthValue() {
        return ethValue;
    }

    public String getDescription() {
        return description;
    }

    @Override
    public String toString() {
        return String.format(
                "Arm{id=%d, name='%s', target='%s', sig='%s', val=%.4f}",
                armId, name, targetContract, functionSig, ethValue
        );
    }
}
