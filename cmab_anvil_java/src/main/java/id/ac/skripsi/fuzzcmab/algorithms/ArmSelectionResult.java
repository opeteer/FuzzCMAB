package id.ac.skripsi.fuzzcmab.algorithms;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Kontainer hasil diagnostik ketika kebijakan CMAB memilih sebuah lengan pada putaran t.
 */
public final class ArmSelectionResult {

    private final int armId;
    private final double ucbScore;
    private final double expectedReward;
    private final double uncertaintyBonus;
    private final Map<Integer, Double> allScores;
    private final Map<String, Object> metadata;

    public ArmSelectionResult(
            int armId,
            double ucbScore,
            double expectedReward,
            double uncertaintyBonus,
            Map<Integer, Double> allScores,
            Map<String, Object> metadata
    ) {
        this.armId = armId;
        this.ucbScore = ucbScore;
        this.expectedReward = expectedReward;
        this.uncertaintyBonus = uncertaintyBonus;
        this.allScores = allScores != null
                ? Collections.unmodifiableMap(new LinkedHashMap<>(allScores))
                : Collections.emptyMap();
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new LinkedHashMap<>(metadata))
                : Collections.emptyMap();
    }

    public int getArmId() {
        return armId;
    }

    public double getUcbScore() {
        return ucbScore;
    }

    public double getExpectedReward() {
        return expectedReward;
    }

    public double getUncertaintyBonus() {
        return uncertaintyBonus;
    }

    public Map<Integer, Double> getAllScores() {
        return allScores;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }
}
