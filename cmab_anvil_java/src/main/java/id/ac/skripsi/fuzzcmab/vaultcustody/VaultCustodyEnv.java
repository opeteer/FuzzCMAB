package id.ac.skripsi.fuzzcmab.vaultcustody;

import id.ac.skripsi.fuzzcmab.common.Arm;
import id.ac.skripsi.fuzzcmab.common.BaseRegularUserBanditEnv;
import id.ac.skripsi.fuzzcmab.common.StepTransition;
import id.ac.skripsi.fuzzcmab.common.RegularUserAnvilClient;
import id.ac.skripsi.fuzzcmab.experiments.GasRegimes;
import id.ac.skripsi.fuzzcmab.util.LinearAlgebra;

import java.math.BigInteger;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Lingkungan CMAB Konkret untuk Agen_Vault_Custody yang berinteraksi dengan
 * VulnerableVault dan MockToken di Anvil sebagai agen pengguna biasa, dilengkapi dengan
 * sistem rezim fluktuasi harga gas dan simulasi transaksi pihak luar (CRN).
 */
public class VaultCustodyEnv extends BaseRegularUserBanditEnv {

    private static final Pattern GAS_USED_REGEX = Pattern.compile("gasUsed\\s+(\\d+|0x[0-9a-fA-F]+)");
    private static final Pattern ADDR_JSON_REGEX = Pattern.compile("\"VulnerableVault\"\\s*:\\s*\"(0x[a-fA-F0-9]{40})\"");
    private static final Pattern TOK_JSON_REGEX = Pattern.compile("\"MockToken\"\\s*:\\s*\"(0x[a-fA-F0-9]{40})\"");

    private final Path deployedJsonPath;
    private final double initialBudgetEth;
    private double remainingBudgetEth;
    private final double minCapitalThresholdEth;
    private final String gasRegime;
    private final int horizonT;
    private final boolean enablePeerTraffic;
    private final long seed;

    private final List<Double> gasSeries;
    private final List<Map<String, Object>> externalEvents;
    private double currentGasGwei;
    private String lastExternalEvent;

    private final PublicContextExtractor contextExtractor;
    private final VaultCustodyRewardOracle rewardOracle;
    private final HttpClient envHttpClient;

    private String vaultAddress = "";
    private String tokenAddress = "";
    private boolean lastSstoreFlag = false;
    private boolean lastWaypointFlag = false;

    public VaultCustodyEnv(
            String rpcUrl,
            Path deployedJsonPath,
            double agentRefBudgetEth,
            String gasRegime,
            int horizonT,
            boolean enablePeerTraffic,
            double minCapitalThresholdEth,
            long seed
    ) {
        super(new RegularUserAnvilClient(rpcUrl), VaultCustodyArms.buildVaultCustodyArmCatalog(), 8);
        this.deployedJsonPath = deployedJsonPath != null
                ? deployedJsonPath
                : resolveDefaultDeployedJsonPath();
        this.initialBudgetEth = agentRefBudgetEth;
        this.remainingBudgetEth = agentRefBudgetEth;
        this.minCapitalThresholdEth = minCapitalThresholdEth;
        this.gasRegime = gasRegime != null ? gasRegime : "stable";
        this.horizonT = Math.max(1, horizonT);
        this.enablePeerTraffic = enablePeerTraffic;
        this.seed = seed;

        this.gasSeries = GasRegimes.generateGasSeries(this.gasRegime, Math.max(this.horizonT, 200), this.seed);
        this.externalEvents = generateExternalPartyEvents(Math.max(this.horizonT, 200), this.seed);
        this.currentGasGwei = this.gasSeries.get(0);
        this.lastExternalEvent = "none";

        this.contextExtractor = new PublicContextExtractor(0.25);
        this.rewardOracle = new VaultCustodyRewardOracle();
        this.envHttpClient = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();

        loadContractAddresses();
    }

    private static Path resolveDefaultDeployedJsonPath() {
        Path curr = Paths.get("").toAbsolutePath();
        for (Path p = curr; p != null; p = p.getParent()) {
            Path candidate = p.resolve("anvil_env").resolve("deployed_addresses.json");
            if (Files.exists(candidate)) {
                return candidate;
            }
        }
        return curr.resolve("anvil_env").resolve("deployed_addresses.json");
    }

    public static List<Map<String, Object>> generateExternalPartyEvents(int rounds, long seed) {
        int n = Math.max(1, rounds);
        Random rng = new Random(seed + 1009L);
        List<Map<String, Object>> events = new ArrayList<>(n);
        for (int t = 1; t <= n; t++) {
            double u = rng.nextDouble();
            String evType;
            if (u < 0.30) {
                evType = "peer_deposit";
            } else if (u < 0.55) {
                evType = "peer_withdraw";
            } else if (u < 0.75) {
                evType = "peer_token_transfer";
            } else {
                evType = "none";
            }
            double ethAmt = Math.round((0.05 + rng.nextDouble() * 0.15) * 1000.0) / 1000.0;
            Map<String, Object> ev = new LinkedHashMap<>();
            ev.put("round", t);
            ev.put("type", evType);
            ev.put("eth_amount", ethAmt);
            events.add(ev);
        }
        return events;
    }

    private void loadContractAddresses() {
        if (!Files.exists(deployedJsonPath)) {
            throw new IllegalStateException(
                    "Berkas deployed_addresses.json tidak ditemukan di " + deployedJsonPath
                            + ". Jalankan ./anvil_env/scripts/run_fair_simulation.sh terlebih dahulu."
            );
        }
        try {
            String raw = Files.readString(deployedJsonPath);
            Matcher mv = ADDR_JSON_REGEX.matcher(raw);
            if (mv.find()) {
                this.vaultAddress = mv.group(1);
            }
            Matcher mt = TOK_JSON_REGEX.matcher(raw);
            if (mt.find()) {
                this.tokenAddress = mt.group(1);
            }
        } catch (Exception e) {
            throw new RuntimeException("Gagal membaca " + deployedJsonPath, e);
        }
    }

    public double getInitialBudgetEth() {
        return initialBudgetEth;
    }

    public double getRemainingBudgetEth() {
        return remainingBudgetEth;
    }

    public double getMinCapitalThresholdEth() {
        return minCapitalThresholdEth;
    }

    public int getHorizonT() {
        return horizonT;
    }

    public String getVaultAddress() {
        return vaultAddress;
    }

    public String getTokenAddress() {
        return tokenAddress;
    }

    @Override
    public Map<String, String> verifyRegularUserStanding() {
        Map<String, String> info = rpcClient.verifyNotOwner(vaultAddress, "owner()(address)");
        Map<String, String> out = new LinkedHashMap<>();
        out.put("agent_name", "Agen_Vault_Custody");
        out.put("agent_address", info.get("agent_address"));
        out.put("vault_owner", info.get("contract_owner"));
        out.put("vault_address", vaultAddress);
        out.put("token_address", tokenAddress);
        out.put("standing", info.get("standing"));
        return out;
    }

    private void setNetworkBaseFee(double gasPriceGwei) {
        this.currentGasGwei = gasPriceGwei;
        long wei = Math.round(gasPriceGwei * 1e9);
        String hexWei = "0x" + Long.toHexString(wei);
        String payload = String.format(
                "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"anvil_setNextBlockBaseFeePerGas\",\"params\":[\"%s\"]}",
                hexWei
        );
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(rpcClient.getRpcUrl()))
                    .timeout(Duration.ofSeconds(3))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8))
                    .build();
            envHttpClient.send(req, HttpResponse.BodyHandlers.discarding());
        } catch (Exception ignored) {
        }
    }

    public static long parseGasUsed(boolean txOk, String txInfo) {
        if (txInfo != null && !txInfo.isBlank()) {
            Matcher m = GAS_USED_REGEX.matcher(txInfo);
            if (m.find()) {
                String raw = m.group(1);
                try {
                    return raw.startsWith("0x") ? Long.parseLong(raw.substring(2), 16) : Long.parseLong(raw);
                } catch (NumberFormatException ignored) {
                }
            }
        }
        return txOk ? 45000L : 28000L;
    }

    @Override
    public OnChainContextSnapshot observeSnapshot() {
        int blockNumber = rpcClient.getBlockNumber();
        double vaultEth = rpcClient.getEthBalance(vaultAddress);
        double agentEoaEth = rpcClient.getEthBalance(rpcClient.getAgentAddress());
        double vaultTotalDep = rpcClient.queryTotalDeposited(vaultAddress, vaultEth);
        double agentVaultBal = rpcClient.queryAgentStake(vaultAddress, rpcClient.getAgentAddress(), null);

        boolean[] solvRes = rpcClient.callBool(vaultAddress, "check_solvency()(bool)");
        boolean solvVal = solvRes[0] ? solvRes[1] : (vaultEth + 1e-9 >= vaultTotalDep);

        boolean[] lockRes = rpcClient.callBool(vaultAddress, "isLocked()(bool)");
        boolean lockedVal = lockRes[0] && lockRes[1];

        boolean codeAlive = rpcClient.getCodeAlive(vaultAddress);
        double agentTokenBal = rpcClient.callUint256EtherWithFallbacks(
                tokenAddress,
                List.of("balanceOf(address)(uint256)"),
                rpcClient.getAgentAddress()
        );

        return new OnChainContextSnapshot(
                blockNumber,
                vaultEth,
                vaultTotalDep,
                agentVaultBal,
                agentEoaEth,
                agentTokenBal,
                solvVal,
                lockedVal,
                codeAlive,
                getRecentRevertRate(),
                lastSstoreFlag,
                lastWaypointFlag,
                contextExtractor.getMinDepositEth()
        );
    }

    @Override
    public double[] observeContextVector(OnChainContextSnapshot snapshot) {
        return contextExtractor.toVector(snapshot);
    }

    @Override
    public RegularUserAnvilClient.TxExecutionResult executeArm(Arm arm) {
        String targetAddr = "VulnerableVault".equals(arm.getTargetContract()) ? vaultAddress : tokenAddress;
        List<String> callArgs = new ArrayList<>();
        String sig = arm.getFunctionSig();
        if ("emergencyDrain(address)".equals(sig) || "emergencyWithdraw(address)".equals(sig)) {
            callArgs.add(rpcClient.getAgentAddress());
        } else if ("mint(address,uint256)".equals(sig)) {
            callArgs.add(rpcClient.getAgentAddress());
            callArgs.add(BigInteger.valueOf(250).multiply(BigInteger.TEN.pow(18)).toString());
        } else if ("transfer(address,uint256)".equals(sig)) {
            callArgs.add(vaultAddress);
            callArgs.add(BigInteger.valueOf(50).multiply(BigInteger.TEN.pow(18)).toString());
        }

        Map<String, String> preStorage = rpcClient.snapshotContractStorage(targetAddr);
        RegularUserAnvilClient.TxExecutionResult res = rpcClient.sendEoaTransaction(
                targetAddr,
                sig,
                callArgs,
                arm.getEthValue(),
                null,
                currentGasGwei
        );

        String txHash = res.isSucceeded() ? rpcClient.extractTxHash(res.getOutput()) : null;
        double[] traceFlags = rpcClient.traceTransactionDataflow(txHash, targetAddr, preStorage);
        this.lastSstoreFlag = traceFlags[0] >= 0.5;
        this.lastWaypointFlag = traceFlags[1] >= 0.5;

        return res;
    }

    @Override
    public BanditRewardBreakdown evaluateReward(
            OnChainContextSnapshot prevSnap,
            OnChainContextSnapshot nextSnap,
            boolean txSucceeded,
            int armId
    ) {
        return rewardOracle.computeReward(prevSnap, nextSnap, txSucceeded, armId);
    }

    public boolean simulatePeerUserDeposit(double ethAmount) {
        return rpcClient.sendEoaTransaction(
                vaultAddress,
                "deposit()",
                null,
                ethAmount,
                RegularUserAnvilClient.PEER_USER_PK,
                currentGasGwei
        ).isSucceeded();
    }

    public boolean simulatePeerUserWithdraw(double ethAmount) {
        double peerStake = rpcClient.queryAgentStake(
                vaultAddress, RegularUserAnvilClient.PEER_USER_ADDR, null
        );
        if (peerStake < ethAmount) {
            return simulatePeerUserDeposit(ethAmount);
        }
        String weiAmt = BigInteger.valueOf(Math.round(ethAmount * 1e9))
                .multiply(BigInteger.valueOf(1_000_000_000L)).toString();
        return rpcClient.sendEoaTransaction(
                vaultAddress,
                "withdraw(uint256)",
                List.of(weiAmt),
                0.0,
                RegularUserAnvilClient.PEER_USER_PK,
                currentGasGwei
        ).isSucceeded();
    }

    public String simulateExternalPartyEvent(int roundIdx) {
        if (!enablePeerTraffic) {
            this.lastExternalEvent = "none";
            return "none";
        }
        int idx = (Math.max(1, roundIdx) - 1) % externalEvents.size();
        Map<String, Object> ev = externalEvents.get(idx);
        String evType = String.valueOf(ev.get("type"));
        double ethAmt = ((Number) ev.get("eth_amount")).doubleValue();

        switch (evType) {
            case "peer_deposit" -> simulatePeerUserDeposit(ethAmt);
            case "peer_withdraw" -> simulatePeerUserWithdraw(ethAmt);
            case "peer_token_transfer" -> rpcClient.sendEoaTransaction(
                    tokenAddress,
                    "mint(address,uint256)",
                    List.of(
                            RegularUserAnvilClient.PEER_USER_ADDR,
                            BigInteger.valueOf(100).multiply(BigInteger.TEN.pow(18)).toString()
                    ),
                    0.0,
                    RegularUserAnvilClient.PEER_USER_PK,
                    currentGasGwei
            );
            default -> {
            }
        }
        this.lastExternalEvent = evType;
        return evType;
    }

    public void prepareRound(int roundIdx) {
        int tIdx = Math.max(0, roundIdx - 1);
        double gT = gasSeries.get(tIdx % gasSeries.size());
        setNetworkBaseFee(gT);
        if (enablePeerTraffic) {
            simulateExternalPartyEvent(tIdx + 1);
        }
    }

    @Override
    public StepTransition step(int armId) {
        return step(armId, null, null);
    }

    public StepTransition step(
            int armId,
            double[] preObservedVec,
            OnChainContextSnapshot preObservedSnap
    ) {
        if (armId < 0 || armId >= arms.size()) {
            throw new IllegalArgumentException("armId tidak valid: " + armId);
        }
        this.roundCounter++;
        Arm armSpec = arms.get(armId);

        OnChainContextSnapshot prevSnap = preObservedSnap != null ? preObservedSnap : observeSnapshot();
        double[] prevVec = preObservedVec != null ? preObservedVec : observeContextVector(prevSnap);

        RegularUserAnvilClient.TxExecutionResult txRes = executeArm(armSpec);
        recordRevertOutcome(txRes.isSucceeded());

        OnChainContextSnapshot nextSnap = observeSnapshot();
        double[] nextVec = observeContextVector(nextSnap);
        BanditRewardBreakdown rewardBreakdown = evaluateReward(prevSnap, nextSnap, txRes.isSucceeded(), armId);

        // Persamaan (3.11): c_t(a_t) = clip((GasUsed_t(a_t) * g_t) + v_{a_t}, c_min, c_max)
        long gasUsed = parseGasUsed(txRes.isSucceeded(), txRes.getOutput());
        double gTEth = currentGasGwei * 1e-9;
        double vAt = armSpec.getEthValue();
        double txCost = LinearAlgebra.clip((gasUsed * gTEth) + vAt, 0.001, 1.0);

        // Persamaan (3.12): B_t = max(0, B_{t-1} - c_t(a_t))
        this.remainingBudgetEth = Math.max(0.0, this.remainingBudgetEth - txCost);

        return new StepTransition(
                roundCounter,
                armSpec,
                txRes.isSucceeded(),
                txRes.getOutput(),
                prevSnap,
                nextSnap,
                prevVec,
                nextVec,
                rewardBreakdown,
                currentGasGwei,
                gasUsed,
                txCost,
                remainingBudgetEth,
                lastExternalEvent
        );
    }
}
