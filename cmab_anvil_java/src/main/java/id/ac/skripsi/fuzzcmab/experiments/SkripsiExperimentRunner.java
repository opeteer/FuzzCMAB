package id.ac.skripsi.fuzzcmab.experiments;

import id.ac.skripsi.fuzzcmab.algorithms.ArmSelectionResult;
import id.ac.skripsi.fuzzcmab.algorithms.BaseCMABPolicy;
import id.ac.skripsi.fuzzcmab.algorithms.DiscountedLinCBwK;
import id.ac.skripsi.fuzzcmab.algorithms.DiscountedLinUCB;
import id.ac.skripsi.fuzzcmab.algorithms.DisjointLinUCB;
import id.ac.skripsi.fuzzcmab.algorithms.DynamicRegretEvaluator;
import id.ac.skripsi.fuzzcmab.algorithms.StationaryLinCBwK;
import id.ac.skripsi.fuzzcmab.algorithms.UniformRandomPolicy;
import id.ac.skripsi.fuzzcmab.common.RegularUserAnvilClient;
import id.ac.skripsi.fuzzcmab.util.LinearAlgebra;
import id.ac.skripsi.fuzzcmab.util.SimpleJson;
import id.ac.skripsi.fuzzcmab.vaultcustody.BanditRewardBreakdown;
import id.ac.skripsi.fuzzcmab.vaultcustody.OnChainContextSnapshot;
import id.ac.skripsi.fuzzcmab.vaultcustody.PublicContextExtractor;
import id.ac.skripsi.fuzzcmab.vaultcustody.VaultCustodyRewardOracle;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Runner Utama Eksperimen Faktorial Multi-Kondisi Skripsi di Java
 * (Porting lengkap dari cmab_anvil/experiments_skripsi/runner.py).
 *
 * Mengevaluasi 5 kebijakan (UniformRandom, LinUCB, D-LinUCB, LinCBwK, D-LinCBwK)
 * pada 4 Tier Kontrak (benchmark, shallow, medium, deep) x 2 Rezim Gas (stable, volatile_spikes)
 * x 3 Variasi Anggaran (tight, moderate, loose) x M Seed Replikasi CRN, lengkap dengan
 * perhitungan 5 Metrik Evaluasi serta Uji Signifikansi Statistik (Paired t-test & Wilcoxon).
 */
public final class SkripsiExperimentRunner {

    private static final String MAX_UINT256 =
            "115792089237316195423570985008687907853269984665640564039457584007913129639935";
    private static final Pattern GAS_USED_REGEX = Pattern.compile("gasUsed\\s+(\\d+|0x[0-9a-fA-F]+)");
    private static final Pattern DEPLOYED_TO_REGEX = Pattern.compile("Deployed to:\\s*(0x[a-fA-F0-9]{40})");
    private static final Pattern CONTRACT_ADDR_REGEX = Pattern.compile("contractAddress\\s+(0x[a-fA-F0-9]{40})");
    private static final VaultCustodyRewardOracle REWARD_ORACLE = new VaultCustodyRewardOracle();
    private static final HttpClient HTTP_CLIENT = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .build();

    private SkripsiExperimentRunner() {
    }

    public static String[] runProcess(List<String> cmd) {
        try {
            ProcessBuilder pb = new ProcessBuilder(cmd);
            Process proc = pb.start();
            StringBuilder out = new StringBuilder();
            StringBuilder err = new StringBuilder();
            try (BufferedReader rOut = new BufferedReader(new InputStreamReader(proc.getInputStream()));
                 BufferedReader rErr = new BufferedReader(new InputStreamReader(proc.getErrorStream()))) {
                String line;
                while ((line = rOut.readLine()) != null) {
                    out.append(line).append('\n');
                }
                while ((line = rErr.readLine()) != null) {
                    err.append(line).append('\n');
                }
            }
            int code = proc.waitFor();
            return new String[]{out.toString().trim(), err.toString().trim(), String.valueOf(code)};
        } catch (Exception e) {
            return new String[]{"", e.getMessage(), "1"};
        }
    }

    public static void ensureAnvilRunning() {
        String[] res = runProcess(List.of(ExperimentConfig.CAST_BIN, "block-number", "--rpc-url", ExperimentConfig.RPC_URL));
        if (!"0".equals(res[2])) {
            System.out.println("[*] Menjalankan instance Anvil di port 8545 (--steps-tracing aktif)...");
            try {
                new ProcessBuilder(
                        ExperimentConfig.ANVIL_BIN,
                        "--port", "8545",
                        "--chain-id", "31337",
                        "--accounts", "10",
                        "--balance", "10000",
                        "--gas-limit", "30000000",
                        "--steps-tracing"
                ).start();
                Thread.sleep(2000L);
            } catch (Exception e) {
                throw new RuntimeException("Gagal menjalankan Anvil", e);
            }
        }
    }

    public static String[] runCast(List<String> args) {
        List<String> cmd = new ArrayList<>();
        cmd.add(ExperimentConfig.CAST_BIN);
        if (!args.isEmpty() && ("keccak".equals(args.get(0)) || "abi-encode".equals(args.get(0)))) {
            cmd.addAll(args);
        } else if (!args.isEmpty()) {
            cmd.add(args.get(0));
            cmd.add("--rpc-url");
            cmd.add(ExperimentConfig.RPC_URL);
            cmd.addAll(args.subList(1, args.size()));
        }
        return runProcess(cmd);
    }

    public static String rpcCall(String method, String paramsJsonArray) {
        String payload = String.format(
                "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"%s\",\"params\":%s}",
                method, paramsJsonArray
        );
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(ExperimentConfig.RPC_URL))
                    .timeout(Duration.ofSeconds(5))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8))
                    .build();
            HttpResponse<String> resp = HTTP_CLIENT.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.body();
        } catch (Exception e) {
            return null;
        }
    }

    public static Path resolveContractPath(ContractCatalog.ContractSpec spec) {
        if ("benchmark".equals(spec.getSplit()) || "VulnerableVault.sol".equals(spec.getFile())) {
            return ExperimentConfig.VULNERABLE_VAULT_SOL;
        }
        return ExperimentConfig.DATASET_DIR.resolve(spec.getSplit()).resolve(spec.getFile());
    }

    public static String compileContract(Path solPath, String contractName) {
        String[] res = runProcess(List.of(
                ExperimentConfig.SOLC_BIN,
                "--optimize",
                "--bin",
                "--via-ir",
                solPath.toString()
        ));
        if (!"0".equals(res[2]) || !res[0].contains("Binary:")) {
            res = runProcess(List.of(
                    ExperimentConfig.SOLC_BIN,
                    "--optimize",
                    "--bin",
                    solPath.toString()
            ));
        }
        if (!res[0].contains("Binary:")) {
            return null;
        }
        String[] lines = res[0].split("\\R");
        String targetBin = null;
        boolean captureNext = false;
        boolean matchTarget = false;

        for (String line : lines) {
            if (line.startsWith("=======") && line.contains(":" + contractName + " =======")) {
                matchTarget = true;
            } else if (line.startsWith("=======")) {
                matchTarget = false;
            }
            if (captureNext) {
                String b = line.trim();
                if (!b.isEmpty()) {
                    if (matchTarget || targetBin == null) {
                        targetBin = b;
                    }
                }
                captureNext = false;
            }
            if ("Binary:".equals(line.trim())) {
                captureNext = true;
            }
        }
        return targetBin != null ? "0x" + targetBin : null;
    }

    public static String deployBytecode(String bytecode, String constructorSig, List<String> constructorArgs) {
        String fullData = bytecode;
        if (constructorSig != null && constructorArgs != null && !constructorArgs.isEmpty()) {
            List<String> encCmd = new ArrayList<>();
            encCmd.add("abi-encode");
            encCmd.add(constructorSig);
            encCmd.addAll(constructorArgs);
            String[] encRes = runCast(encCmd);
            if ("0".equals(encRes[2]) && encRes[0].startsWith("0x")) {
                fullData = bytecode + encRes[0].substring(2);
            }
        }
        String[] res = runCast(List.of("send", "--private-key", ExperimentConfig.DEPLOYER_PK, "--create", fullData));
        if (!"0".equals(res[2])) {
            return null;
        }
        Matcher m1 = DEPLOYED_TO_REGEX.matcher(res[0]);
        if (m1.find()) {
            return m1.group(1);
        }
        Matcher m2 = CONTRACT_ADDR_REGEX.matcher(res[0]);
        if (m2.find()) {
            return m2.group(1);
        }
        return null;
    }

    public static String[] sendTx(
            String pk,
            String toAddr,
            String sig,
            List<String> args,
            double valueEth,
            Double gasPriceGwei
    ) {
        List<String> cmd = new ArrayList<>();
        cmd.add("send");
        cmd.add("--private-key");
        cmd.add(pk);
        if (valueEth > 0.0) {
            BigInteger wei = BigInteger.valueOf(Math.round(valueEth * 1e9))
                    .multiply(BigInteger.valueOf(1_000_000_000L));
            cmd.add("--value");
            cmd.add(wei.toString());
        }
        if (gasPriceGwei != null && gasPriceGwei > 0.0) {
            long weiGas = Math.round(gasPriceGwei * 1e9);
            cmd.add("--gas-price");
            cmd.add(String.valueOf(weiGas));
        }
        cmd.add(toAddr);
        if (sig != null && !sig.isEmpty()) {
            cmd.add(sig);
            if (args != null && !args.isEmpty()) {
                cmd.addAll(args);
            }
        }
        return runCast(cmd);
    }

    public static long parseGasUsed(boolean txOk, String txOut, int nArgs) {
        if (txOut != null && !txOut.isEmpty()) {
            Matcher m = GAS_USED_REGEX.matcher(txOut);
            if (m.find()) {
                String raw = m.group(1);
                try {
                    return raw.startsWith("0x") ? Long.parseLong(raw.substring(2), 16) : Long.parseLong(raw);
                } catch (NumberFormatException ignored) {
                }
            }
        }
        return txOk ? (45000L + 6000L * nArgs) : (28000L + 2000L * nArgs);
    }

    public static String computeSingleNodeMerkleRoot(String accountAddr, BigInteger amountWei) {
        String cleanAddr = accountAddr.toLowerCase().replace("0x", "");
        cleanAddr = "0".repeat(Math.max(0, 64 - cleanAddr.length())) + cleanAddr;
        String cleanAmt = amountWei.toString(16);
        cleanAmt = "0".repeat(Math.max(0, 64 - cleanAmt.length())) + cleanAmt;
        String hexConcat = cleanAddr + cleanAmt;
        byte[] encoded = new byte[64];
        for (int i = 0; i < 64; i++) {
            encoded[i] = (byte) Integer.parseInt(hexConcat.substring(2 * i, 2 * i + 2), 16);
        }
        byte[] leaf = RegularUserAnvilClient.keccak256Bytes(encoded);
        StringBuilder sb = new StringBuilder("0x");
        for (byte b : leaf) {
            sb.append(String.format("%02x", b & 0xFF));
        }
        return sb.toString();
    }

    public static double getEscrowBalance(String caddr, RegularUserAnvilClient client) {
        return client.callUint256EtherWithFallbacks(
                caddr,
                List.of("payments(address)(uint256)"),
                ExperimentConfig.DEPLOYER_ADDR
        );
    }

    public static boolean checkShuffleRoom2Intact(String caddr) {
        String[] res = runCast(List.of("call", caddr, "getRoom(uint256)", "2"));
        if (!"0".equals(res[2]) || res[0].isEmpty()) {
            return true;
        }
        String agentClean = ExperimentConfig.AGENT_ADDR.toLowerCase().replace("0x", "");
        return !res[0].toLowerCase().contains(agentClean);
    }

    public static Map<String, Object> setupMemberEnvironment(
            ContractCatalog.ContractSpec spec,
            String caddr,
            String mockTokenAddr,
            RegularUserAnvilClient client
    ) {
        String cid = spec.getId();
        String depPk = ExperimentConfig.DEPLOYER_PK;
        String peerPk = ExperimentConfig.PEER_PK;
        String agentPk = ExperimentConfig.AGENT_PK;

        sendTx(depPk, mockTokenAddr, "mint(address,uint256)", List.of(ExperimentConfig.AGENT_ADDR, "100000000000000000000000"), 0.0, null);
        sendTx(depPk, mockTokenAddr, "mint(address,uint256)", List.of(ExperimentConfig.PEER_ADDR, "100000000000000000000000"), 0.0, null);
        sendTx(depPk, mockTokenAddr, "mint(address,uint256)", List.of(caddr, "100000000000000000000000"), 0.0, null);
        sendTx(agentPk, mockTokenAddr, "approve(address,uint256)", List.of(caddr, MAX_UINT256), 0.0, null);
        sendTx(peerPk, mockTokenAddr, "approve(address,uint256)", List.of(caddr, MAX_UINT256), 0.0, null);

        String custodyTokenAddr = mockTokenAddr;

        if (cid.contains("B01")) {
            sendTx(depPk, caddr, "deposit()", List.of(), 4.48, null);
            sendTx(peerPk, caddr, "deposit()", List.of(), 0.50, null);
        } else if (cid.contains("S01")) {
            custodyTokenAddr = caddr;
            sendTx(depPk, caddr, "enableTrading()", List.of(), 0.0, null);
            sendTx(depPk, caddr, "setDepositMinMax(uint256,uint256)", List.of("10000000000000000", "100000000000000000000"), 0.0, null);
            sendTx(depPk, caddr, "transfer(address,uint256)", List.of(ExperimentConfig.AGENT_ADDR, "50000000000000000000"), 0.0, null);
            sendTx(depPk, caddr, "transfer(address,uint256)", List.of(ExperimentConfig.PEER_ADDR, "50000000000000000000"), 0.0, null);
            sendTx(depPk, caddr, "transfer(address,uint256)", List.of(caddr, "10000000000000000000"), 0.0, null);
            sendTx(agentPk, caddr, "approve(address,uint256)", List.of(caddr, MAX_UINT256), 0.0, null);
            sendTx(peerPk, caddr, "approve(address,uint256)", List.of(caddr, MAX_UINT256), 0.0, null);
            sendTx(depPk, caddr, "", List.of(), 1.0, null);
            sendTx(peerPk, caddr, "", List.of(), 0.50, null);
        } else if (cid.contains("S02")) {
            custodyTokenAddr = caddr;
            String agentBalSlot = RegularUserAnvilClient.computeSolidityMappingSlot(ExperimentConfig.AGENT_ADDR, 1);
            String peerBalSlot = RegularUserAnvilClient.computeSolidityMappingSlot(ExperimentConfig.PEER_ADDR, 1);
            String hexVal = BigInteger.valueOf(1000).multiply(BigInteger.TEN.pow(18)).toString(16);
            String initTokHex = "0x" + "0".repeat(Math.max(0, 64 - hexVal.length())) + hexVal;
            rpcCall("anvil_setStorageAt", String.format("[\"%s\",\"%s\",\"%s\"]", caddr, agentBalSlot, initTokHex));
            rpcCall("anvil_setStorageAt", String.format("[\"%s\",\"%s\",\"%s\"]", caddr, peerBalSlot, initTokHex));
            sendTx(agentPk, caddr, "approve(address,uint256)", List.of(caddr, MAX_UINT256), 0.0, null);
        } else if (cid.contains("M01")) {
            sendTx(depPk, caddr, "setCost(uint256)", List.of("10000000000000000"), 0.0, null);
            sendTx(depPk, caddr, "mint(string)", List.of("STATIC_DUPLICATE"), 0.0, null);
        } else if (cid.contains("M02")) {
            sendTx(depPk, caddr, "createRoom(string,uint256,uint256,uint256,uint256,uint256,address)",
                    List.of("Room1", "10000000000000000", "5000000000000000000", "5", "10", "3600", ExperimentConfig.DEPLOYER_ADDR), 0.0, null);
            sendTx(depPk, caddr, "createRoom(string,uint256,uint256,uint256,uint256,uint256,address)",
                    List.of("Room2", "10000000000000000", "5000000000000000000", "5", "10", "3600", ExperimentConfig.DEPLOYER_ADDR), 0.0, null);
        } else if (cid.contains("D01")) {
            custodyTokenAddr = caddr;
            sendTx(depPk, caddr, "transfer(address,uint256)", List.of(ExperimentConfig.AGENT_ADDR, "5000000000000000000000"), 0.0, null);
            sendTx(depPk, caddr, "transfer(address,uint256)", List.of(ExperimentConfig.PEER_ADDR, "5000000000000000000000"), 0.0, null);
            sendTx(depPk, caddr, "transfer(address,uint256)", List.of(caddr, "100000000000000000000"), 0.0, null);
            sendTx(agentPk, caddr, "approve(address,uint256)", List.of(caddr, MAX_UINT256), 0.0, null);
            sendTx(peerPk, caddr, "approve(address,uint256)", List.of(caddr, MAX_UINT256), 0.0, null);
        } else if (cid.contains("D02")) {
            sendTx(depPk, caddr, "initialize(address,address)", List.of(mockTokenAddr, ExperimentConfig.DEPLOYER_ADDR), 0.0, null);
            sendTx(depPk, caddr, "setAllowStaking(bool)", List.of("true"), 0.0, null);
            sendTx(depPk, caddr, "setAllowClaim(bool)", List.of("true"), 0.0, null);
            sendTx(depPk, caddr, "setReward(uint256,uint256)", List.of("1", "5000"), 0.0, null);
            sendTx(depPk, caddr, "setReward(uint256,uint256)", List.of("7", "8000"), 0.0, null);
        } else if (cid.contains("D03")) {
            sendTx(depPk, caddr, "setToken(address)", List.of(mockTokenAddr), 0.0, null);
            String merkleRoot = computeSingleNodeMerkleRoot(
                    ExperimentConfig.AGENT_ADDR,
                    BigInteger.valueOf(250).multiply(BigInteger.TEN.pow(18))
            );
            sendTx(depPk, caddr, "setRoot(bytes32)", List.of(merkleRoot), 0.0, null);
        }

        double initEth = client.getEthBalance(caddr);
        double initEscrow = cid.contains("M01") ? getEscrowBalance(caddr, client) : 0.0;
        double initTokRes = client.callUint256EtherWithFallbacks(
                custodyTokenAddr, List.of("balanceOf(address)(uint256)"), caddr
        );

        Map<String, Object> state = new LinkedHashMap<>();
        state.put("cid", cid);
        state.put("custody_token_addr", custodyTokenAddr);
        state.put("eth_liability", initEth + initEscrow);
        state.put("tok_liability", initTokRes);
        state.put("peer_deposited_eth", (cid.contains("B01") || cid.contains("S01")) ? 0.50 : 0.0);
        return state;
    }

    public static List<Map<String, Object>> pregenerateNetworkEvents(int contractIdx, int rounds, int seedOffset) {
        Random rng = new Random(202600L + contractIdx * 37L + seedOffset * 1009L);
        List<Map<String, Object>> events = new ArrayList<>(rounds);
        for (int t = 1; t <= rounds; t++) {
            double u = rng.nextDouble();
            String evType;
            if (u < 0.25) {
                evType = "peer_liquidity";
            } else if (u < 0.50) {
                evType = "peer_withdrawal";
            } else if (u < 0.70) {
                evType = "block_jitter";
            } else if (u < 0.85) {
                evType = "congestion_poke";
            } else {
                evType = "none";
            }
            double ethDelta = Math.round((0.02 + rng.nextDouble() * 0.06) * 1000.0) / 1000.0;
            int timeDelta = 3600 + rng.nextInt(864000 - 3600);
            Map<String, Object> ev = new LinkedHashMap<>();
            ev.put("round", t);
            ev.put("type", evType);
            ev.put("eth_delta", ethDelta);
            ev.put("time_delta", timeDelta);
            events.add(ev);
        }
        return events;
    }

    public static void executeNetworkEvent(
            Map<String, Object> ev,
            String caddr,
            String mockTokenAddr,
            Map<String, Object> envState,
            RegularUserAnvilClient client
    ) {
        String cid = String.valueOf(envState.get("cid"));
        String evType = String.valueOf(ev.get("type"));
        double ethDelta = ((Number) ev.get("eth_delta")).doubleValue();
        int timeDelta = ((Number) ev.get("time_delta")).intValue();

        switch (evType) {
            case "peer_liquidity" -> {
                if (cid.contains("B01")) {
                    double curEth = client.getEthBalance(caddr);
                    if (curEth + ethDelta < 4.99) {
                        String[] r = sendTx(ExperimentConfig.PEER_PK, caddr, "deposit()", List.of(), ethDelta, null);
                        if ("0".equals(r[2])) {
                            envState.put("peer_deposited_eth", ((Number) envState.get("peer_deposited_eth")).doubleValue() + ethDelta);
                            envState.put("eth_liability", ((Number) envState.get("eth_liability")).doubleValue() + ethDelta);
                        }
                    }
                } else if (cid.contains("S01")) {
                    String[] r = sendTx(ExperimentConfig.PEER_PK, caddr, "", List.of(), ethDelta, null);
                    if ("0".equals(r[2])) {
                        envState.put("peer_deposited_eth", ((Number) envState.get("peer_deposited_eth")).doubleValue() + ethDelta);
                        envState.put("eth_liability", ((Number) envState.get("eth_liability")).doubleValue() + ethDelta);
                    }
                } else {
                    sendTx(ExperimentConfig.PEER_PK, mockTokenAddr, "transfer(address,uint256)",
                            List.of(ExperimentConfig.DEPLOYER_ADDR, "1000000000000000"), 0.0, null);
                }
            }
            case "peer_withdrawal" -> {
                double peerDep = ((Number) envState.getOrDefault("peer_deposited_eth", 0.0)).doubleValue();
                if (cid.contains("B01") && peerDep >= ethDelta) {
                    double curEth = client.getEthBalance(caddr);
                    if (curEth - ethDelta >= 4.94) {
                        String weiStr = BigInteger.valueOf(Math.round(ethDelta * 1e9))
                                .multiply(BigInteger.valueOf(1_000_000_000L)).toString();
                        String[] r = sendTx(ExperimentConfig.PEER_PK, caddr, "withdraw(uint256)", List.of(weiStr), 0.0, null);
                        if ("0".equals(r[2])) {
                            envState.put("peer_deposited_eth", peerDep - ethDelta);
                            double liab = ((Number) envState.get("eth_liability")).doubleValue();
                            envState.put("eth_liability", Math.max(0.0, liab - ethDelta));
                        }
                    } else {
                        sendTx(ExperimentConfig.PEER_PK, mockTokenAddr, "transfer(address,uint256)",
                                List.of(ExperimentConfig.DEPLOYER_ADDR, "500000000000000"), 0.0, null);
                    }
                } else {
                    sendTx(ExperimentConfig.PEER_PK, mockTokenAddr, "transfer(address,uint256)",
                            List.of(ExperimentConfig.DEPLOYER_ADDR, "500000000000000"), 0.0, null);
                }
            }
            case "block_jitter" -> {
                rpcCall("evm_increaseTime", "[" + timeDelta + "]");
                rpcCall("evm_mine", "[]");
            }
            case "congestion_poke" -> sendTx(ExperimentConfig.PEER_PK, mockTokenAddr, "transfer(address,uint256)",
                    List.of(ExperimentConfig.DEPLOYER_ADDR, "1000000000000000"), 0.0, null);
            default -> {
            }
        }
    }

    public static OnChainContextSnapshot buildSnapshot(
            String caddr,
            RegularUserAnvilClient client,
            Map<String, Object> envState,
            double recentRevertRate,
            double agentBudgetEth,
            boolean sstoreExecuted,
            boolean dataflowWaypoint,
            double minDepositRequirement
    ) {
        String cid = String.valueOf(envState.get("cid"));
        String custodyTokAddr = String.valueOf(envState.get("custody_token_addr"));

        int block = client.getBlockNumber();
        double vaultEth = client.getEthBalance(caddr);
        double escrowEth = cid.contains("M01") ? getEscrowBalance(caddr, client) : 0.0;
        double effectiveVaultEth = vaultEth + escrowEth;

        boolean codeAlive = client.getCodeAlive(caddr);
        if (cid.contains("M02") && codeAlive) {
            codeAlive = checkShuffleRoom2Intact(caddr);
        }

        double agentBal = client.queryAgentStake(caddr, ExperimentConfig.AGENT_ADDR, custodyTokAddr);
        double tokenBal = client.callUint256EtherWithFallbacks(
                custodyTokAddr, List.of("balanceOf(address)(uint256)"), ExperimentConfig.AGENT_ADDR
        );
        double contractTokRes = client.callUint256EtherWithFallbacks(
                custodyTokAddr, List.of("balanceOf(address)(uint256)"), caddr
        );

        double totalDep = ((Number) envState.get("eth_liability")).doubleValue();
        double tokLiab = ((Number) envState.get("tok_liability")).doubleValue();
        boolean ethSolvent = (effectiveVaultEth + 1e-6 >= totalDep);
        boolean tokSolvent = (contractTokRes + 1e-6 >= tokLiab);
        boolean isSolvent = ethSolvent && tokSolvent;

        boolean[] lockRes = client.callBool(caddr, "isLocked()(bool)");
        boolean isLocked = lockRes[0] && lockRes[1];

        return new OnChainContextSnapshot(
                block,
                effectiveVaultEth,
                totalDep,
                agentBal,
                agentBudgetEth,
                tokenBal,
                isSolvent,
                isLocked,
                codeAlive,
                recentRevertRate,
                sstoreExecuted,
                dataflowWaypoint,
                minDepositRequirement
        );
    }

    public static Object[] executeMemberArm(
            ContractCatalog.CatalogArmSpec arm,
            String caddr,
            String mockTokenAddr,
            int roundT,
            double gasPriceGwei
    ) {
        String target = "MOCK_TOKEN".equals(arm.getTarget()) ? mockTokenAddr : caddr;
        String sig = arm.getSig();
        double val = arm.getVal();

        List<String> resolvedArgs = new ArrayList<>();
        for (String a : arm.getArgs()) {
            switch (a) {
                case "DYNAMIC_URI" -> resolvedArgs.add("member_nft_" + roundT);
                case "DYNAMIC_URI_LOW" -> resolvedArgs.add("member_low_" + roundT);
                case "DYNAMIC_SIG65" -> resolvedArgs.add(String.format("0x%064x", roundT) + "00".repeat(33));
                case "DYNAMIC_STAKE_IDX" -> {
                    String[] stRes = runCast(List.of("call", caddr, "getTotalStakes()(uint256)"));
                    int idxVal = 0;
                    if ("0".equals(stRes[2]) && !stRes[0].isEmpty()) {
                        try {
                            String tok = stRes[0].split("\\s+")[0];
                            int totalSt = tok.startsWith("0x")
                                    ? Integer.parseInt(tok.substring(2), 16)
                                    : Integer.parseInt(tok);
                            idxVal = Math.max(0, totalSt - 1);
                        } catch (Exception ignored) {
                        }
                    }
                    resolvedArgs.add(String.valueOf(idxVal));
                }
                case "MOCK_TOKEN" -> resolvedArgs.add(mockTokenAddr);
                default -> resolvedArgs.add(a);
            }
        }

        String[] txRes = sendTx(ExperimentConfig.AGENT_PK, target, sig, resolvedArgs, val, gasPriceGwei);
        boolean ok = "0".equals(txRes[2]);
        String out = ok ? txRes[0] : txRes[1];
        return new Object[]{ok, out, resolvedArgs};
    }

    public static void updateLegitimateLiabilitiesBeforePostSnapshot(
            ContractCatalog.CatalogArmSpec arm,
            boolean txOk,
            String caddr,
            Map<String, Object> envState,
            RegularUserAnvilClient client
    ) {
        if (!txOk) {
            return;
        }
        String cid = String.valueOf(envState.get("cid"));
        String sig = arm.getSig();
        double val = arm.getVal();

        if (val > 0.0) {
            envState.put("eth_liability", ((Number) envState.get("eth_liability")).doubleValue() + val);
        }
        if ((cid.contains("B01") && "withdraw()".equals(sig))
                || (cid.contains("M02") && sig.startsWith("withdrawBid"))) {
            double curEth = client.getEthBalance(caddr);
            double liab = ((Number) envState.get("eth_liability")).doubleValue();
            envState.put("eth_liability", Math.min(liab, curEth));
        }

        if (cid.contains("S01") && sig.startsWith("deposit(uint256)")) {
            double amtTok = new BigInteger(arm.getArgs().get(0)).doubleValue() / 1e18;
            envState.put("tok_liability", ((Number) envState.get("tok_liability")).doubleValue() + amtTok);
        } else if (cid.contains("D01") && sig.startsWith("stake(uint256,uint256)")) {
            double amtTok = new BigInteger(arm.getArgs().get(0)).doubleValue() / 1e18;
            envState.put("tok_liability", ((Number) envState.get("tok_liability")).doubleValue() + amtTok);
            rpcCall("evm_increaseTime", "[2600000]");
            rpcCall("evm_mine", "[]");
        } else if (cid.contains("D02") && sig.startsWith("stake(uint256,uint256)")) {
            double amtTok = new BigInteger(arm.getArgs().get(0)).doubleValue() / 1e18;
            envState.put("tok_liability", ((Number) envState.get("tok_liability")).doubleValue() + amtTok);
            rpcCall("evm_increaseTime", "[604800]");
            rpcCall("evm_mine", "[]");
        } else if (cid.contains("D02") && sig.startsWith("unlockPremature")) {
            String custodyTokAddr = String.valueOf(envState.get("custody_token_addr"));
            double curTok = client.callUint256EtherWithFallbacks(
                    custodyTokAddr, List.of("balanceOf(address)(uint256)"), caddr
            );
            double liab = ((Number) envState.get("tok_liability")).doubleValue();
            envState.put("tok_liability", Math.min(liab, curTok));
        } else if (cid.contains("M02") && sig.startsWith("placeBid(uint256)")) {
            rpcCall("evm_increaseTime", "[90000]");
            rpcCall("evm_mine", "[]");
        }
    }

    public static void reanchorPostVulnerabilityState(
            String caddr,
            Map<String, Object> envState,
            RegularUserAnvilClient client,
            BanditRewardBreakdown breakdown
    ) {
        String cid = String.valueOf(envState.get("cid"));
        String custodyTokAddr = String.valueOf(envState.get("custody_token_addr"));

        if ("E4".equals(breakdown.getEventPartition())) {
            double curEth = client.getEthBalance(caddr);
            if (cid.contains("B01") && curEth < 0.01) {
                String zeroHex = "0x" + "0".repeat(64);
                for (String addr : List.of(ExperimentConfig.AGENT_ADDR, ExperimentConfig.PEER_ADDR, ExperimentConfig.DEPLOYER_ADDR)) {
                    String slot = RegularUserAnvilClient.computeSolidityMappingSlot(addr, 0);
                    rpcCall("anvil_setStorageAt", String.format("[\"%s\",\"%s\",\"%s\"]", caddr, slot, zeroHex));
                }
                rpcCall("anvil_setStorageAt", String.format("[\"%s\",\"0x1\",\"%s\"]", caddr, zeroHex));
                sendTx(ExperimentConfig.DEPLOYER_PK, caddr, "deposit()", List.of(), 4.48, null);
                sendTx(ExperimentConfig.PEER_PK, caddr, "deposit()", List.of(), 0.50, null);
                envState.put("peer_deposited_eth", 0.50);
            } else if (cid.contains("M02") && !checkShuffleRoom2Intact(caddr)) {
                sendTx(ExperimentConfig.DEPLOYER_PK, caddr,
                        "modifyRoom(uint256,bool,uint256,address)",
                        List.of("2", "true", "5", ExperimentConfig.DEPLOYER_ADDR), 0.0, null);
            } else if (cid.contains("D03")) {
                String merkleRoot = computeSingleNodeMerkleRoot(
                        ExperimentConfig.AGENT_ADDR,
                        BigInteger.valueOf(250).multiply(BigInteger.TEN.pow(18))
                );
                sendTx(ExperimentConfig.DEPLOYER_PK, caddr, "setRoot(bytes32)", List.of(merkleRoot), 0.0, null);
            }
        }

        double vaultEth = client.getEthBalance(caddr);
        double escrowEth = cid.contains("M01") ? getEscrowBalance(caddr, client) : 0.0;
        envState.put("eth_liability", vaultEth + escrowEth);

        double contractTokRes = client.callUint256EtherWithFallbacks(
                custodyTokAddr, List.of("balanceOf(address)(uint256)"), caddr
        );
        envState.put("tok_liability", contractTokRes);
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> runSinglePolicy(
            String policyName,
            BaseCMABPolicy policy,
            ContractCatalog.ContractSpec spec,
            String contractBytecode,
            String mockTokenBytecode,
            List<Map<String, Object>> networkEvents,
            List<Double> gasPrices,
            double initialBudget,
            int horizonT
    ) {
        String mockTokenAddr = deployBytecode(
                mockTokenBytecode, "constructor(uint256)", List.of("1000000000000000000000000")
        );
        String caddr = deployBytecode(
                contractBytecode, spec.getConstructorSig(), spec.getConstructorArgs()
        );
        if (caddr == null || mockTokenAddr == null) {
            throw new RuntimeException("Gagal men-deploy kontrak " + spec.getContract());
        }

        RegularUserAnvilClient client = new RegularUserAnvilClient(ExperimentConfig.RPC_URL);
        Map<String, Object> envState = setupMemberEnvironment(spec, caddr, mockTokenAddr, client);

        BigInteger eoaFundWei = BigInteger.valueOf(Math.round(Math.max(10.0, initialBudget * 5.0) * 1e9))
                .multiply(BigInteger.valueOf(1_000_000_000L));
        rpcCall("anvil_setBalance", String.format("[\"%s\",\"0x%s\"]",
                ExperimentConfig.AGENT_ADDR, eoaFundWei.toString(16)));
        double remainingBudget = initialBudget;

        List<ContractCatalog.CatalogArmSpec> arms = spec.getArms();
        int k = arms.size();
        double minDepReq = 0.05;
        for (ContractCatalog.CatalogArmSpec a : arms) {
            if (a.getVal() > 0.0 && a.getVal() < minDepReq) {
                minDepReq = a.getVal();
            }
        }
        PublicContextExtractor extractor = new PublicContextExtractor(minDepReq);

        List<Double> rewards = new ArrayList<>();
        List<Double> costs = new ArrayList<>();
        List<Double> decisionTimesMs = new ArrayList<>();
        List<Map<String, Object>> roundLogs = new ArrayList<>();
        int[] armCounts = new int[k];
        Map<String, Integer> eventCounts = new LinkedHashMap<>();
        for (String p : List.of("E0", "E1", "E2", "E3", "E4")) {
            eventCounts.put(p, 0);
        }

        int successCount = 0;
        int sstoreCount = 0;
        int waypointCount = 0;
        int vulnTriggerCount = 0;
        List<String> anomaliesFound = new ArrayList<>();
        List<Integer> recentReverts = new ArrayList<>();
        boolean lastSstoreFlag = false;
        boolean lastWaypointFlag = false;
        String terminationReason = "completed";

        for (int t = 1; t <= horizonT; t++) {
            if (remainingBudget <= ExperimentConfig.MIN_CAPITAL_THRESHOLD_ETH) {
                terminationReason = "bankruptcy";
                break;
            }

            double currentGasGwei = (t - 1 < gasPrices.size()) ? gasPrices.get(t - 1) : 20.0;
            long baseFeeWei = Math.round(currentGasGwei * 1e9);
            rpcCall("anvil_setNextBlockBaseFeePerGas", String.format("[\"0x%s\"]", Long.toHexString(baseFeeWei)));

            if (t - 1 < networkEvents.size()) {
                executeNetworkEvent(networkEvents.get(t - 1), caddr, mockTokenAddr, envState, client);
            }

            double revRate = 0.0;
            if (!recentReverts.isEmpty()) {
                int start = Math.max(0, recentReverts.size() - 8);
                int sumRev = 0;
                for (int i = start; i < recentReverts.size(); i++) {
                    sumRev += recentReverts.get(i);
                }
                revRate = (double) sumRev / (double) (recentReverts.size() - start);
            }

            OnChainContextSnapshot prevSnap = buildSnapshot(
                    caddr, client, envState, revRate, remainingBudget,
                    lastSstoreFlag, lastWaypointFlag, minDepReq
            );
            double[] vec = extractor.toVector(prevSnap);

            long t0Dec = System.nanoTime();
            ArmSelectionResult sel = policy.selectArm(vec);
            long t1Dec = System.nanoTime();

            int armIdx = sel.getArmId();
            ContractCatalog.CatalogArmSpec chosenArm = arms.get(armIdx);
            armCounts[armIdx]++;

            String targetForTrace = "MOCK_TOKEN".equals(chosenArm.getTarget()) ? mockTokenAddr : caddr;
            Map<String, String> preStorage = client.snapshotContractStorage(targetForTrace);
            double preAgentEoaEth = client.getEthBalance(ExperimentConfig.AGENT_ADDR);

            Object[] execRes = executeMemberArm(chosenArm, caddr, mockTokenAddr, t, currentGasGwei);
            boolean txOk = (Boolean) execRes[0];
            String txOut = (String) execRes[1];
            List<String> resolvedArgs = (List<String>) execRes[2];

            if (txOk) {
                successCount++;
            }
            recentReverts.add(txOk ? 0 : 1);

            String txHash = txOk ? client.extractTxHash(txOut) : null;
            double[] traceFlags = client.traceTransactionDataflow(txHash, targetForTrace, preStorage);
            lastSstoreFlag = traceFlags[0] >= 0.5;
            lastWaypointFlag = traceFlags[1] >= 0.5;
            if (lastSstoreFlag) {
                sstoreCount++;
            }
            if (lastWaypointFlag) {
                waypointCount++;
            }

            updateLegitimateLiabilitiesBeforePostSnapshot(chosenArm, txOk, caddr, envState, client);

            double postAgentEoaEth = client.getEthBalance(ExperimentConfig.AGENT_ADDR);
            double eoaEthDelta = postAgentEoaEth - preAgentEoaEth;

            OnChainContextSnapshot nextSnap = buildSnapshot(
                    caddr, client, envState, revRate,
                    remainingBudget + Math.max(0.0, eoaEthDelta),
                    lastSstoreFlag, lastWaypointFlag, minDepReq
            );

            BanditRewardBreakdown breakdown = REWARD_ORACLE.computeReward(prevSnap, nextSnap, txOk, armIdx);
            double rew = breakdown.getTotalReward();
            String evPart = breakdown.getEventPartition();
            eventCounts.put(evPart, eventCounts.getOrDefault(evPart, 0) + 1);
            if ("E4".equals(evPart)) {
                vulnTriggerCount++;
            }

            reanchorPostVulnerabilityState(caddr, envState, client, breakdown);

            long gasUsed = parseGasUsed(txOk, txOut, resolvedArgs.size());
            double gTEth = currentGasGwei * 1e-9;
            double vAt = chosenArm.getVal();
            double txCost = LinearAlgebra.clip((gasUsed * gTEth) + vAt, 0.001, 1.0);
            remainingBudget = Math.max(0.0, remainingBudget - txCost);

            long t2Dec = System.nanoTime();
            policy.update(armIdx, vec, rew, txCost);
            long t3Dec = System.nanoTime();

            double overheadMs = ((t1Dec - t0Dec) + (t3Dec - t2Dec)) / 1_000_000.0;
            decisionTimesMs.add(overheadMs);
            rewards.add(rew);
            costs.add(txCost);
            anomaliesFound.addAll(breakdown.getDetectedAnomalies());

            String evName = (t - 1 < networkEvents.size())
                    ? String.valueOf(networkEvents.get(t - 1).get("type"))
                    : "none";

            double cumRewNow = 0.0;
            for (double r : rewards) {
                cumRewNow += r;
            }
            double burnedNow = 0.0;
            for (double c : costs) {
                burnedNow += c;
            }

            Map<String, Object> roundLog = new LinkedHashMap<>();
            roundLog.put("round", t);
            roundLog.put("gas_price_gwei", currentGasGwei);
            roundLog.put("gas_used", gasUsed);
            roundLog.put("arm_id", armIdx);
            roundLog.put("arm_name", chosenArm.getName());
            roundLog.put("arm_role", chosenArm.getRole());
            roundLog.put("tx_ok", txOk);
            roundLog.put("event_partition", evPart);
            roundLog.put("sstore_executed", lastSstoreFlag);
            roundLog.put("dataflow_waypoint", lastWaypointFlag);
            roundLog.put("reward", rew);
            roundLog.put("cost_eth", txCost);
            roundLog.put("cumulative_reward", cumRewNow);
            roundLog.put("balance_eth", remainingBudget);
            roundLog.put("burned_eth", burnedNow);
            roundLog.put("dual_lambda", policy.getLambdaT());
            roundLog.put("decision_overhead_ms", overheadMs);
            roundLog.put("event_type", evName);
            roundLogs.add(roundLog);

            if (remainingBudget <= ExperimentConfig.MIN_CAPITAL_THRESHOLD_ETH) {
                terminationReason = "bankruptcy";
                break;
            }
        }

        int actualRounds = rewards.size();
        double capitalBurned = 0.0;
        for (double c : costs) {
            capitalBurned += c;
        }
        double cumReward = 0.0;
        for (double r : rewards) {
            cumReward += r;
        }
        double effGas = cumReward / Math.max(1e-6, capitalBurned);
        double survivalRate = (double) actualRounds / (double) Math.max(1, horizonT);

        double meanOverheadMs = 0.0;
        if (!decisionTimesMs.isEmpty()) {
            for (double d : decisionTimesMs) {
                meanOverheadMs += d;
            }
            meanOverheadMs /= decisionTimesMs.size();
        }

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("policy", policyName);
        out.put("contract", spec.getContract());
        out.put("rounds_survived", actualRounds);
        out.put("budget_survival_rate", survivalRate);
        out.put("termination_reason", terminationReason);
        out.put("bankruptcy", "bankruptcy".equals(terminationReason));
        out.put("initial_budget_eth", initialBudget);
        out.put("final_capital_eth", remainingBudget);
        out.put("capital_burned_eth", capitalBurned);
        out.put("gas_efficiency_ratio", effGas);
        out.put("cumulative_reward", cumReward);
        out.put("mean_reward", actualRounds > 0 ? cumReward / actualRounds : 0.0);
        out.put("vuln_trigger_count", vulnTriggerCount);
        out.put("sstore_count", sstoreCount);
        out.put("waypoint_count", waypointCount);
        out.put("event_counts", eventCounts);
        out.put("success_rate", (double) successCount / (double) Math.max(1, actualRounds));
        out.put("arm_counts", armCounts);
        out.put("final_dual_lambda", policy.getLambdaT());
        out.put("mean_execution_overhead_ms", meanOverheadMs);
        out.put("anomalies", new ArrayList<>(new LinkedHashSet<>(anomaliesFound)));
        out.put("round_logs", roundLogs);
        return out;
    }

    public static Map<String, Object> runExperimentMatrix(
            String tier,
            String regime,
            String budget,
            int horizon,
            Integer contractsLimit,
            double alpha,
            double gamma,
            int seeds
    ) throws Exception {
        ensureAnvilRunning();

        List<String> tiers = "all".equals(tier)
                ? List.of("benchmark", "shallow", "medium", "deep")
                : List.of(tier);
        List<String> regimes = "all".equals(regime)
                ? List.of("stable", "volatile_spikes")
                : List.of(regime);
        List<String> budgets = "all".equals(budget)
                ? List.of("tight", "moderate", "loose")
                : List.of(budget);

        String mockTokenBin = compileContract(ExperimentConfig.MOCK_TOKEN_SOL, "MockToken");
        if (mockTokenBin == null) {
            throw new RuntimeException("Gagal mengompilasi MockToken.sol");
        }

        List<Map<String, Object>> allExperiments = new ArrayList<>();
        long tStart = System.currentTimeMillis();

        System.out.println("=".repeat(104));
        System.out.printf(Locale.US,
                "MEMULAI EKSPERIMEN SKRIPSI FAKTORIAL (JAVA) | T = %d Putaran | Alpha = %.2f | Gamma = %.2f | Seeds M = %d%n",
                horizon, alpha, gamma, seeds);
        System.out.println("   Kondisi 1 (Tiers)  : " + tiers);
        System.out.println("   Kondisi 2 (Regimes): " + regimes);
        System.out.println("   Kondisi 3 (Budgets): " + budgets);
        System.out.println("=".repeat(104));

        for (String tName : tiers) {
            List<ContractCatalog.ContractSpec> contracts = ContractCatalog.getContractsByTier(tName);
            if (contractsLimit != null && contractsLimit > 0 && contractsLimit < contracts.size()) {
                contracts = contracts.subList(0, contractsLimit);
            }

            for (ContractCatalog.ContractSpec cSpec : contracts) {
                String cname = cSpec.getContract();
                int cIdx = cSpec.getContractIdx();
                Path fpath = resolveContractPath(cSpec);
                System.out.printf("%n[+] Kompilasi Kontrak: %s (Tier: %s | %s)%n",
                        cname, tName.toUpperCase(), cSpec.getFile());
                String cBin = compileContract(fpath, cname);
                if (cBin == null) {
                    System.out.println("    [!] Gagal kompilasi " + cname + "; dilewati.");
                    continue;
                }

                for (String regName : regimes) {
                    for (String bName : budgets) {
                        double initB = ExperimentConfig.BUDGET_TIERS.get(bName);
                        for (int seedIdx = 0; seedIdx < seeds; seedIdx++) {
                            long crnSeed = 2026L + seedIdx * 101L;
                            List<Double> gasSeries = GasRegimes.generateGasSeries(
                                    regName, horizon, 30300L + cIdx * 41L + seedIdx * 997L
                            );
                            List<Map<String, Object>> netEvents = pregenerateNetworkEvents(cIdx, horizon, seedIdx);

                            System.out.printf(Locale.US,
                                    "%n---> MENGEKSEKUSI: %s | Rezim: %s | Anggaran: %s (%.2f ETH)%n",
                                    cname, regName.toUpperCase(), bName.toUpperCase(), initB);

                            int k = cSpec.getArms().size();
                            Map<String, BaseCMABPolicy> policies = new LinkedHashMap<>();
                            policies.put("UniformRandom", new UniformRandomPolicy(
                                    k, ExperimentConfig.CONTEXT_DIM, initB, horizon, crnSeed, "UniformRandom"
                            ));
                            policies.put("LinUCB", new DisjointLinUCB(
                                    k, ExperimentConfig.CONTEXT_DIM, alpha, ExperimentConfig.L2_LAMBDA, crnSeed
                            ));
                            policies.put("D-LinUCB", new DiscountedLinUCB(
                                    k, ExperimentConfig.CONTEXT_DIM, gamma, alpha, ExperimentConfig.L2_LAMBDA,
                                    false, false, crnSeed
                            ));
                            policies.put("LinCBwK", new StationaryLinCBwK(
                                    k, ExperimentConfig.CONTEXT_DIM, initB, horizon, alpha,
                                    ExperimentConfig.L2_LAMBDA, ExperimentConfig.INITIAL_LAMBDA, crnSeed, "LinCBwK"
                            ));
                            policies.put("D-LinCBwK", DiscountedLinCBwK.createDefault(
                                    k, ExperimentConfig.CONTEXT_DIM, gamma, initB, horizon, alpha,
                                    ExperimentConfig.L2_LAMBDA, ExperimentConfig.INITIAL_LAMBDA, crnSeed, "D-LinCBwK"
                            ));

                            Map<String, Object> cellResults = new LinkedHashMap<>();
                            Map<String, List<Map<String, Object>>> policiesRoundLogs = new LinkedHashMap<>();

                            for (Map.Entry<String, BaseCMABPolicy> pEntry : policies.entrySet()) {
                                String pTag = pEntry.getKey();
                                Map<String, Object> res = runSinglePolicy(
                                        pTag, pEntry.getValue(), cSpec, cBin, mockTokenBin,
                                        netEvents, gasSeries, initB, horizon
                                );
                                cellResults.put(pTag, res);
                                @SuppressWarnings("unchecked")
                                List<Map<String, Object>> rLogs = (List<Map<String, Object>>) res.get("round_logs");
                                policiesRoundLogs.put(pTag, rLogs);
                            }

                            Map<String, DynamicRegretEvaluator.DynamicRegretSummary> regretSummaries =
                                    DynamicRegretEvaluator.comparePoliciesDynamicRegret(
                                            policiesRoundLogs,
                                            ExperimentConfig.CONTEXT_DIM,
                                            k,
                                            initB,
                                            1.00,
                                            horizon
                                    );

                            for (Map.Entry<String, DynamicRegretEvaluator.DynamicRegretSummary> rEntry : regretSummaries.entrySet()) {
                                String pTag = rEntry.getKey();
                                DynamicRegretEvaluator.DynamicRegretSummary regSum = rEntry.getValue();
                                @SuppressWarnings("unchecked")
                                Map<String, Object> res = (Map<String, Object>) cellResults.get(pTag);
                                res.put("cumulative_dynamic_regret", regSum.getCumulativeDynamicRegret());
                                res.put("mean_dynamic_regret", regSum.getMeanDynamicRegret());
                                res.put("state_switches_S", regSum.getStateSwitches());

                                System.out.printf(Locale.US,
                                        "     * %-14s: SR=%5.1f%% (%2d/%d) | Rew=%5.2f | D-Regret=%5.2f | Vuln(E4)=%2d | Eff_gas=%6.2f | Overhead=%.3fms%n",
                                        pTag,
                                        ((Number) res.get("budget_survival_rate")).doubleValue() * 100.0,
                                        ((Number) res.get("rounds_survived")).intValue(),
                                        horizon,
                                        ((Number) res.get("cumulative_reward")).doubleValue(),
                                        regSum.getCumulativeDynamicRegret(),
                                        ((Number) res.get("vuln_trigger_count")).intValue(),
                                        ((Number) res.get("gas_efficiency_ratio")).doubleValue(),
                                        ((Number) res.get("mean_execution_overhead_ms")).doubleValue()
                                );
                            }

                            Map<String, Object> expRecord = new LinkedHashMap<>();
                            expRecord.put("contract_id", cSpec.getId());
                            expRecord.put("contract_name", cname);
                            expRecord.put("tier", tName);
                            expRecord.put("gas_regime", regName);
                            expRecord.put("budget_tier", bName);
                            expRecord.put("seed_idx", seedIdx);
                            expRecord.put("crn_seed", crnSeed);
                            expRecord.put("initial_budget_eth", initB);
                            expRecord.put("horizon_t", horizon);
                            expRecord.put("results", cellResults);
                            allExperiments.add(expRecord);
                        }
                    }
                }
            }
        }

        double elapsedSec = (System.currentTimeMillis() - tStart) / 1000.0;
        String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        Files.createDirectories(ExperimentConfig.TELEMETRY_DIR);
        Path outFile = ExperimentConfig.TELEMETRY_DIR.resolve("skripsi_experiments_java_" + ts + ".json");
        Path latestFile = ExperimentConfig.TELEMETRY_DIR.resolve("skripsi_experiments_java_latest.json");

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("timestamp", ts);
        payload.put("elapsed_seconds", elapsedSec);
        payload.put("total_experiments", allExperiments.size());
        payload.put("experiments", allExperiments);

        String jsonText = SimpleJson.toPrettyJson(payload);
        Files.writeString(outFile, jsonText);
        Files.writeString(latestFile, jsonText);

        System.out.println("\n" + "=".repeat(104));
        System.out.printf(Locale.US, "EKSPERIMEN SKRIPSI JAVA SELESAI DALAM %.2f DETIK%n", elapsedSec);
        System.out.println("   Telemetri Disimpan ke: " + outFile);
        System.out.println("=".repeat(104));
        return payload;
    }

    public static void main(String[] args) throws Exception {
        String tier = "all";
        String regime = "all";
        String budget = "all";
        int horizon = ExperimentConfig.DEFAULT_HORIZON_T;
        Integer contractsLimit = null;
        double alpha = ExperimentConfig.ALPHA;
        double gamma = ExperimentConfig.GAMMA;
        int seeds = 1;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--tier" -> tier = args[++i];
                case "--regime" -> regime = args[++i];
                case "--budget" -> budget = args[++i];
                case "--horizon" -> horizon = Integer.parseInt(args[++i]);
                case "--contracts-limit" -> contractsLimit = Integer.parseInt(args[++i]);
                case "--alpha" -> alpha = Double.parseDouble(args[++i]);
                case "--gamma" -> gamma = Double.parseDouble(args[++i]);
                case "--seeds" -> seeds = Integer.parseInt(args[++i]);
                default -> {
                }
            }
        }

        runExperimentMatrix(tier, regime, budget, horizon, contractsLimit, alpha, gamma, seeds);
    }
}
