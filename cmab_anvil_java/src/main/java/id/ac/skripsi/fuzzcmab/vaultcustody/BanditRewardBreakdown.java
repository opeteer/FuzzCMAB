package id.ac.skripsi.fuzzcmab.vaultcustody;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Dekomposisi modular imbalan skalar r_t in {0.00, 0.20, 0.50, 0.70, 1.00} (Tabel 3.2 Bab III).
 */
public final class BanditRewardBreakdown {

    private final double totalReward;
    private final double rExec;
    private final double rState;
    private final double rEcon;
    private final double rVuln;
    private final String eventPartition;
    private final List<String> detectedAnomalies;

    public BanditRewardBreakdown(
            double totalReward,
            double rExec,
            double rState,
            double rEcon,
            double rVuln,
            String eventPartition,
            List<String> detectedAnomalies
    ) {
        this.totalReward = totalReward;
        this.rExec = rExec;
        this.rState = rState;
        this.rEcon = rEcon;
        this.rVuln = rVuln;
        this.eventPartition = eventPartition != null ? eventPartition : "E0";
        this.detectedAnomalies = detectedAnomalies != null
                ? Collections.unmodifiableList(new ArrayList<>(detectedAnomalies))
                : Collections.emptyList();
    }

    public double getTotalReward() {
        return totalReward;
    }

    public double getRExec() {
        return rExec;
    }

    public double getRState() {
        return rState;
    }

    public double getREcon() {
        return rEcon;
    }

    public double getRVuln() {
        return rVuln;
    }

    public String getEventPartition() {
        return eventPartition;
    }

    public List<String> getDetectedAnomalies() {
        return detectedAnomalies;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("total_reward", totalReward);
        map.put("r_exec", rExec);
        map.put("r_state", rState);
        map.put("r_econ", rEcon);
        map.put("r_vuln", rVuln);
        map.put("event_partition", eventPartition);
        map.put("detected_anomalies", detectedAnomalies);
        return map;
    }
}
