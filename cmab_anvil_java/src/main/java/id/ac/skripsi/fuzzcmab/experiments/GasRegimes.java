package id.ac.skripsi.fuzzcmab.experiments;

import id.ac.skripsi.fuzzcmab.util.LinearAlgebra;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Modul Pembangkitan Dinamika Harga Gas EVM (Kondisi 2 Eksperimen Skripsi - Tabel 3.1 Bab III):
 *   - Rezim 1 (stable): Log-normal EIP-1559 (mu = ln(15), sigma = 0.35), clipped [8.0, 32.0] Gwei
 *   - Rezim 2 (volatile_spikes): Uniform [60, 140] Gwei + 25% lonjakan Pareto (a = 1.8 * 80), clipped [60.0, 500.0] Gwei
 *   - Kontrol (static): Konstan 1.0 Gwei
 */
public final class GasRegimes {

    private GasRegimes() {
    }

    public static List<Double> generateGasSeries(String regime, int rounds, long seed) {
        int n = Math.max(1, rounds);
        String reg = regime != null ? regime.toLowerCase().trim() : "stable";
        List<Double> series = new ArrayList<>(n);

        if ("static".equals(reg)) {
            for (int i = 0; i < n; i++) {
                series.add(1.0);
            }
            return series;
        }

        Random rng = new Random(seed);
        if (reg.equals("stable") || reg.equals("low_volatility") || reg.equals("normal")) {
            double mu = Math.log(15.0);
            double sigma = 0.35;
            for (int i = 0; i < n; i++) {
                double sample = Math.exp(mu + sigma * rng.nextGaussian());
                double clipped = LinearAlgebra.clip(Math.round(sample * 100.0) / 100.0, 8.0, 32.0);
                series.add(clipped);
            }
            return series;
        }

        if (reg.equals("volatile_spikes") || reg.equals("high_volatility") || reg.equals("worst_case")) {
            for (int i = 0; i < n; i++) {
                double base = 60.0 + rng.nextDouble() * 80.0;
                boolean hasSpike = rng.nextDouble() < 0.25;
                double spike = 0.0;
                if (hasSpike) {
                    double u = Math.max(1e-9, rng.nextDouble());
                    double pareto = (1.0 / Math.pow(u, 1.0 / 1.8)) - 1.0;
                    spike = pareto * 80.0;
                }
                double clipped = LinearAlgebra.clip(Math.round((base + spike) * 100.0) / 100.0, 60.0, 500.0);
                series.add(clipped);
            }
            return series;
        }

        throw new IllegalArgumentException(
                "Rezim gas tidak dikenal: '" + regime + "'. Pilih 'stable', 'volatile_spikes', atau 'static'."
        );
    }
}
