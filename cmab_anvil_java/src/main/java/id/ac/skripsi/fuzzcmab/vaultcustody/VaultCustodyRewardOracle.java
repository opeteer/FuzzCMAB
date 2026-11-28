package id.ac.skripsi.fuzzcmab.vaultcustody;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Pemodelan Fungsi Imbalan Berjenjang 4-Komponen (Persamaan 3.3 - 3.5 & Tabel 3.2 Bab III):
 *     r_t(tau_t, I_t) = I(tau_t = 1) * min(1.00, w^T * I_t)
 *
 * Dengan vektor bobot w = [0.20, 0.30, 0.20, 1.00]^T dan himpunan support:
 *     supp(r_t) = {0.00, 0.20, 0.50, 0.70, 1.00} (Partisi kejadian E0, E1, E2, E3, E4).
 */
public class VaultCustodyRewardOracle {

    public static final double[] REWARD_SUPPORT_LEVELS = {0.00, 0.20, 0.50, 0.70, 1.00};
    public static final double[] REWARD_WEIGHT_VECTOR = {0.20, 0.30, 0.20, 1.00};
    public static final double ASSET_GAIN_EPS = 0.001;

    public BanditRewardBreakdown computeReward(
            OnChainContextSnapshot prevSnap,
            OnChainContextSnapshot nextSnap,
            boolean txSucceeded,
            int armId
    ) {
        // 1. Kejadian E0 (Level 0): Transaksi Gagal / Reverted di EVM (tau_t = 0) -> v_0 = 0.00
        if (!txSucceeded) {
            return new BanditRewardBreakdown(
                    0.00, 0.0, 0.0, 0.0, 0.0, "E0", Collections.emptyList()
            );
        }

        List<String> anomalies = new ArrayList<>();

        // 2. Komponen 1: E_exec,t = I(tau_t = 1), w_1 = 0.20
        double rExec = 0.20;

        // 3. Komponen 2: E_state,t = I(Delta S_storage,t != 0), w_2 = 0.30
        boolean hasStateMutation = nextSnap.isSstoreExecuted()
                || Math.abs(nextSnap.getAgentVaultBalance() - prevSnap.getAgentVaultBalance()) > 1e-6
                || Math.abs(nextSnap.getVaultTotalDeposited() - prevSnap.getVaultTotalDeposited()) > 1e-6;
        double rState = hasStateMutation ? 0.30 : 0.0;

        // 4. Komponen 3: E_econ,t = I(Delta S_storage,t != 0 AND Delta Aset_eoa,t > eps), w_3 = 0.20
        double deltaEoaEth = nextSnap.getAgentEoaEthBalance() - prevSnap.getAgentEoaEthBalance();
        double deltaToken = nextSnap.getAgentTokenBalance() - prevSnap.getAgentTokenBalance();
        boolean gainedAssets = (deltaEoaEth > ASSET_GAIN_EPS) || (deltaToken > ASSET_GAIN_EPS);
        boolean eEcon = hasStateMutation && gainedAssets;
        double rEcon = eEcon ? 0.20 : 0.0;

        if (eEcon && deltaEoaEth > ASSET_GAIN_EPS) {
            anomalies.add(String.format(Locale.US, "ETH_EXTRACTION (+%.3f ETH)", deltaEoaEth));
        }
        if (eEcon && deltaToken > ASSET_GAIN_EPS) {
            anomalies.add(String.format(Locale.US, "TOKEN_ACQUISITION (+%.0f Tokens)", deltaToken));
        }

        // 5. Komponen 4: E_vuln,t = I((B_t >= L_t AND B_{t+1} < L_{t+1}) OR Omega_t > Omega_{t+1}), w_4 = 1.00
        boolean solvencyBreached = prevSnap.isVaultIsSolvent() && !nextSnap.isVaultIsSolvent();
        boolean contractKilled = prevSnap.isVaultCodeAlive() && !nextSnap.isVaultCodeAlive();

        double rVuln = 0.0;
        if (solvencyBreached) {
            rVuln = 1.00;
            anomalies.add("SOLVENCY_INVARIANT_BREACH (vault_balance < totalDeposited)");
        }
        if (contractKilled) {
            rVuln = 1.00;
            anomalies.add("CONTRACT_SELFDESTRUCTED (code destroyed)");
        }

        // Persamaan (3.4): r_t = I(tau_t = 1) * min(1.00, w^T * I_t)
        double total = Math.round(Math.min(1.00, rExec + rState + rEcon + rVuln) * 100.0) / 100.0;

        String evPart;
        if (rVuln >= 1.00) {
            evPart = "E4";
        } else if (eEcon) {
            evPart = "E3";
        } else if (hasStateMutation) {
            evPart = "E2";
        } else {
            evPart = "E1";
        }

        return new BanditRewardBreakdown(total, rExec, rState, rEcon, rVuln, evPart, anomalies);
    }
}
