package id.ac.skripsi.fuzzcmab.algorithms;

/**
 * Kebijakan 4: Stationary LinUCB with Knapsacks (LinCBwK).
 *
 * Reduksi ablasi stasioner dari D-LinCBwK (Sub-bab 3.3.5) dengan menetapkan faktor diskonto
 * gamma = 1.0 sambil mempertahankan pembaruan variabel dual Knapsack lambda_t (Persamaan 3.13).
 */
public class StationaryLinCBwK extends DiscountedLinCBwK {

    public StationaryLinCBwK(
            int nArms,
            int contextDim,
            double initialBudget,
            int totalHorizon,
            double alpha,
            double l2Lambda,
            double initialLambda,
            long seed,
            String name
    ) {
        super(
                nArms,
                contextDim,
                initialBudget,
                totalHorizon,
                1.0,
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
                name != null ? name : "LinCBwK"
        );
    }
}
