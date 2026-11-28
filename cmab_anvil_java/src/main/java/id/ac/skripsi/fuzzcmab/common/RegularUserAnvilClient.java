package id.ac.skripsi.fuzzcmab.common;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Klien JSON-RPC & Foundry Cast untuk Agen Pengguna Biasa (Regular User Anvil Client).
 *
 * Menegakkan Model Agen Pengguna Biasa:
 *   1. Menggunakan identitas EOA pengguna biasa (DEFAULT_AGENT_ADDR).
 *   2. Memverifikasi secara on-chain bahwa agen BUKAN owner kontrak.
 *   3. Melarang pemanggilan cheatcode manipulasi state (anvil_*, evm_*, hardhat_*).
 *   4. Mengimplementasikan Keccak-256 Ethereum murni (0x01 padding) untuk inferensi slot mapping
 *      Solidity keccak256(abi.encode(user, p)) serta pelacakan opcode SLOAD/SSTORE hibrida.
 */
public class RegularUserAnvilClient {

    public static final String DEPLOYER_ADDR = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266";
    public static final String PEER_USER_ADDR = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8";
    public static final String PEER_USER_PK = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";
    public static final String DEFAULT_AGENT_ADDR = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC";
    public static final String DEFAULT_AGENT_PK = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a";

    private static final List<String> FORBIDDEN_RPC_PREFIXES = List.of("anvil_", "evm_", "hardhat_");
    private static final Pattern TX_HASH_REGEX = Pattern.compile("0x[a-fA-F0-9]{64}");
    private static final Pattern SLOAD_SSTORE_REGEX = Pattern.compile(
            "\"op\"\\s*:\\s*\"(SLOAD|SSTORE)\"[\\s\\S]*?\"stack\"\\s*:\\s*\\[([^\\]]*)\\]"
    );
    private static final Pattern HEX_TOKEN_REGEX = Pattern.compile("\"(0x[a-fA-F0-9]+)\"");

    public static final List<String> STANDARD_GLOBAL_LIABILITY_SIGS = List.of(
            "totalDeposited()(uint256)",
            "totalAssets()(uint256)",
            "totalStaked()(uint256)",
            "totalSupply()(uint256)",
            "poolBalance()(uint256)",
            "totalBalance()(uint256)",
            "getTotalDeposits()(uint256)"
    );

    public static final List<String> STANDARD_USER_STAKE_SIGS = List.of(
            "balances(address)(uint256)",
            "balanceOf(address)(uint256)",
            "stakes(address)(uint256)",
            "deposits(address)(uint256)",
            "shares(address)(uint256)",
            "userInfo(address)(uint256)",
            "getStake(address)(uint256)"
    );

    // Konstanta Keccak-256 (24 putaran)
    private static final long[] KECCAK_RC = {
            0x0000000000000001L, 0x0000000000008082L, 0x800000000000808AL, 0x8000000080008000L,
            0x000000000000808BL, 0x0000000080000001L, 0x8000000080008081L, 0x8000000000008009L,
            0x000000000000008AL, 0x0000000000000088L, 0x0000000080008009L, 0x000000008000000AL,
            0x000000008000808BL, 0x800000000000008BL, 0x8000000000008089L, 0x8000000000008003L,
            0x8000000000008002L, 0x8000000000000080L, 0x000000000000800AL, 0x800000008000000AL,
            0x8000000080008081L, 0x8000000000008080L, 0x0000000080000001L, 0x8000000080008008L
    };

    private static final int[][] KECCAK_ROT = {
            {0, 36, 3, 41, 18},
            {1, 44, 10, 45, 2},
            {62, 6, 43, 15, 61},
            {28, 55, 25, 21, 56},
            {27, 20, 39, 8, 14}
    };

    public static final class TxExecutionResult {
        private final boolean succeeded;
        private final String output;

        public TxExecutionResult(boolean succeeded, String output) {
            this.succeeded = succeeded;
            this.output = output != null ? output : "";
        }

        public boolean isSucceeded() {
            return succeeded;
        }

        public String getOutput() {
            return output;
        }
    }

    private final String rpcUrl;
    private final String agentAddress;
    private final String agentPrivateKey;
    private final Path castBin;
    private final int numTrackedSlots;
    private final HttpClient httpClient;

    private final Map<String, Set<String>> observedSloadSlots = new HashMap<>();
    private final Map<String, Map<String, String>> lastStorageSnapshot = new HashMap<>();
    private final Map<String, List<String>> dynamicGlobalGetters = new HashMap<>();
    private final Map<String, List<String>> dynamicUserGetters = new HashMap<>();
    private final Map<String, Integer> inferredMappingSlot = new HashMap<>();

    public RegularUserAnvilClient(String rpcUrl) {
        this(rpcUrl, DEFAULT_AGENT_ADDR, DEFAULT_AGENT_PK,
                Paths.get(System.getProperty("user.home"), ".foundry", "bin", "cast"), 8);
    }

    public RegularUserAnvilClient(
            String rpcUrl,
            String agentAddress,
            String agentPrivateKey,
            Path castBin,
            int numTrackedSlots
    ) {
        this.rpcUrl = rpcUrl != null ? rpcUrl : "http://127.0.0.1:8545";
        this.agentAddress = agentAddress != null ? agentAddress : DEFAULT_AGENT_ADDR;
        this.agentPrivateKey = agentPrivateKey != null ? agentPrivateKey : DEFAULT_AGENT_PK;
        this.castBin = castBin != null
                ? castBin
                : Paths.get(System.getProperty("user.home"), ".foundry", "bin", "cast");
        this.numTrackedSlots = Math.max(1, numTrackedSlots);
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
    }

    public String getRpcUrl() {
        return rpcUrl;
    }

    public String getAgentAddress() {
        return agentAddress;
    }

    public String getAgentPrivateKey() {
        return agentPrivateKey;
    }

    /**
     * Menghitung hash Ethereum Keccak-256 (0x01 padding) secara murni di Java.
     */
    public static byte[] keccak256Bytes(byte[] data) {
        final int rate = 136;
        int paddedLen = ((data.length + 1 + rate - 1) / rate) * rate;
        if (paddedLen == 0) {
            paddedLen = rate;
        }
        byte[] padded = Arrays.copyOf(data, paddedLen);
        padded[data.length] = 0x01;
        padded[paddedLen - 1] |= (byte) 0x80;

        long[][] state = new long[5][5];
        for (int offset = 0; offset < paddedLen; offset += rate) {
            for (int i = 0; i < rate / 8; i++) {
                int x = i % 5;
                int y = i / 5;
                long lane = 0L;
                for (int b = 0; b < 8; b++) {
                    lane |= ((long) (padded[offset + 8 * i + b] & 0xFF)) << (8 * b);
                }
                state[x][y] ^= lane;
            }
            for (int r = 0; r < 24; r++) {
                long[] c = new long[5];
                for (int x = 0; x < 5; x++) {
                    c[x] = state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4];
                }
                long[] d = new long[5];
                for (int x = 0; x < 5; x++) {
                    d[x] = c[(x + 4) % 5] ^ Long.rotateLeft(c[(x + 1) % 5], 1);
                }
                for (int x = 0; x < 5; x++) {
                    for (int y = 0; y < 5; y++) {
                        state[x][y] ^= d[x];
                    }
                }
                long[][] bMat = new long[5][5];
                for (int x = 0; x < 5; x++) {
                    for (int y = 0; y < 5; y++) {
                        bMat[y][(2 * x + 3 * y) % 5] = Long.rotateLeft(state[x][y], KECCAK_ROT[x][y]);
                    }
                }
                for (int x = 0; x < 5; x++) {
                    for (int y = 0; y < 5; y++) {
                        state[x][y] = bMat[x][y] ^ ((~bMat[(x + 1) % 5][y]) & bMat[(x + 2) % 5][y]);
                    }
                }
                state[0][0] ^= KECCAK_RC[r];
            }
        }

        byte[] out = new byte[32];
        int pos = 0;
        for (int y = 0; y < 5 && pos < 32; y++) {
            for (int x = 0; x < 5 && pos < 32; x++) {
                long lane = state[x][y];
                for (int b = 0; b < 8 && pos < 32; b++) {
                    out[pos++] = (byte) ((lane >>> (8 * b)) & 0xFF);
                }
            }
        }
        return out;
    }

    /**
     * Menghitung slot storage deterministik Solidity untuk mapping(address => uint256) pada slot dasar p:
     *     keccak256(abi.encode(address, uint256(p)))
     */
    public static String computeSolidityMappingSlot(String userAddress, int mappingBaseSlot) {
        String cleanAddr = userAddress.toLowerCase().replace("0x", "");
        cleanAddr = "0".repeat(Math.max(0, 64 - cleanAddr.length())) + cleanAddr;
        String cleanSlot = Integer.toHexString(mappingBaseSlot);
        cleanSlot = "0".repeat(Math.max(0, 64 - cleanSlot.length())) + cleanSlot;
        String hexConcat = cleanAddr + cleanSlot;

        byte[] encoded = new byte[64];
        for (int i = 0; i < 64; i++) {
            encoded[i] = (byte) Integer.parseInt(hexConcat.substring(2 * i, 2 * i + 2), 16);
        }
        byte[] hash = keccak256Bytes(encoded);
        StringBuilder sb = new StringBuilder("0x");
        for (byte b : hash) {
            sb.append(String.format("%02x", b & 0xFF));
        }
        return sb.toString();
    }

    private void checkForbiddenMethod(String token) {
        for (String forbidden : FORBIDDEN_RPC_PREFIXES) {
            if (token.startsWith(forbidden)) {
                throw new SecurityException(
                        "Pelanggaran Batas Agen Pengguna Biasa: Cheatcode manipulasi state '" + token + "' diblokir!"
                );
            }
        }
    }

    public String jsonRpcRaw(String method, String paramsJsonArray) {
        checkForbiddenMethod(method);
        String payload = String.format(
                "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"%s\",\"params\":%s}",
                method, paramsJsonArray
        );
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(rpcUrl))
                    .timeout(Duration.ofSeconds(5))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8))
                    .build();
            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.body();
        } catch (Exception e) {
            return null;
        }
    }

    public String[] runCast(List<String> args) {
        for (String token : args) {
            checkForbiddenMethod(token);
        }
        List<String> cmd = new ArrayList<>();
        cmd.add(castBin.toString());
        cmd.addAll(args);
        cmd.add("--rpc-url");
        cmd.add(rpcUrl);

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

    public static double parseUint256Ether(String rawOutput) {
        if (rawOutput == null || rawOutput.isBlank()) {
            return 0.0;
        }
        String firstToken = rawOutput.trim().split("\\s+")[0];
        try {
            BigInteger val = firstToken.startsWith("0x") || firstToken.startsWith("0X")
                    ? new BigInteger(firstToken.substring(2), 16)
                    : new BigInteger(firstToken);
            return val.doubleValue() / 1e18;
        } catch (Exception e) {
            return 0.0;
        }
    }

    public int getBlockNumber() {
        String[] res = runCast(List.of("block-number"));
        if (!"0".equals(res[2]) || res[0].isEmpty()) {
            return 0;
        }
        try {
            String tok = res[0].split("\\s+")[0];
            return tok.startsWith("0x") ? Integer.parseInt(tok.substring(2), 16) : Integer.parseInt(tok);
        } catch (Exception e) {
            return 0;
        }
    }

    public double getEthBalance(String address) {
        String[] res = runCast(List.of("balance", address));
        if (!"0".equals(res[2])) {
            return 0.0;
        }
        return parseUint256Ether(res[0]);
    }

    public boolean getCodeAlive(String address) {
        String[] res = runCast(List.of("code", address));
        return "0".equals(res[2]) && res[0].length() > 2 && !"0x".equalsIgnoreCase(res[0]);
    }

    public String getStorageAt(String contractAddress, String slotHexOrIdx) {
        String slotHex = slotHexOrIdx;
        if (!slotHex.startsWith("0x")) {
            try {
                slotHex = "0x" + Integer.toHexString(Integer.parseInt(slotHex));
            } catch (NumberFormatException ignored) {
            }
        }
        String[] res = runCast(List.of("storage", contractAddress, slotHex));
        if ("0".equals(res[2]) && !res[0].isEmpty()) {
            return res[0].split("\\s+")[0].toLowerCase();
        }
        return "0x" + "0".repeat(64);
    }

    public double callUint256Ether(String contractAddress, String sig, String... callArgs) {
        List<String> cmd = new ArrayList<>();
        cmd.add("call");
        cmd.add(contractAddress);
        cmd.add(sig);
        Collections.addAll(cmd, callArgs);
        String[] res = runCast(cmd);
        if (!"0".equals(res[2])) {
            return 0.0;
        }
        return parseUint256Ether(res[0]);
    }

    public double callUint256EtherWithFallbacks(
            String contractAddress,
            List<String> candidateSigs,
            String... callArgs
    ) {
        for (String sig : candidateSigs) {
            double val = callUint256Ether(contractAddress, sig, callArgs);
            if (val > 0.0) {
                return val;
            }
        }
        if (callArgs.length == 1 && callArgs[0].startsWith("0x") && callArgs[0].length() == 42) {
            double mapped = inferMappingBalanceFromStorage(contractAddress, callArgs[0]);
            if (mapped > 0.0) {
                return mapped;
            }
        }
        return 0.0;
    }

    public boolean[] callBool(String contractAddress, String sig, String... callArgs) {
        List<String> cmd = new ArrayList<>();
        cmd.add("call");
        cmd.add(contractAddress);
        cmd.add(sig);
        Collections.addAll(cmd, callArgs);
        String[] res = runCast(cmd);
        if (!"0".equals(res[2])) {
            return new boolean[]{false, false};
        }
        return new boolean[]{true, res[0].toLowerCase().contains("true")};
    }

    public double inferGlobalLiabilityFromStorage(String contractAddress, double vaultEthHint) {
        List<Double> candidates = new ArrayList<>();
        BigInteger minWei = BigInteger.TEN.pow(14);
        BigInteger maxWei = BigInteger.TEN.pow(26);
        for (int slotIdx = 0; slotIdx < numTrackedSlots; slotIdx++) {
            String rawHex = getStorageAt(contractAddress, "0x" + Integer.toHexString(slotIdx));
            try {
                BigInteger rawInt = new BigInteger(rawHex.replace("0x", ""), 16);
                if (rawInt.compareTo(minWei) >= 0 && rawInt.compareTo(maxWei) <= 0) {
                    candidates.add(rawInt.doubleValue() / 1e18);
                }
            } catch (Exception ignored) {
            }
        }
        if (candidates.isEmpty()) {
            return 0.0;
        }
        if (vaultEthHint > 0.0) {
            candidates.sort((a, b) -> Double.compare(Math.abs(a - vaultEthHint), Math.abs(b - vaultEthHint)));
        }
        return candidates.get(0);
    }

    public double inferMappingBalanceFromStorage(String contractAddress, String userAddress) {
        String uAddr = userAddress != null ? userAddress : agentAddress;
        String addrKey = contractAddress.toLowerCase();
        BigInteger minWei = BigInteger.TEN.pow(12);
        BigInteger maxWei = BigInteger.TEN.pow(27);

        if (inferredMappingSlot.containsKey(addrKey)) {
            int pCached = inferredMappingSlot.get(addrKey);
            String slotHash = computeSolidityMappingSlot(uAddr, pCached);
            String rawHex = getStorageAt(contractAddress, slotHash);
            try {
                BigInteger rawInt = new BigInteger(rawHex.replace("0x", ""), 16);
                if (rawInt.compareTo(BigInteger.ZERO) > 0 && rawInt.compareTo(maxWei) <= 0) {
                    return rawInt.doubleValue() / 1e18;
                }
            } catch (Exception ignored) {
            }
        }

        for (int p = 0; p < numTrackedSlots; p++) {
            String slotHash = computeSolidityMappingSlot(uAddr, p);
            String rawHex = getStorageAt(contractAddress, slotHash);
            try {
                BigInteger rawInt = new BigInteger(rawHex.replace("0x", ""), 16);
                if (rawInt.compareTo(minWei) >= 0 && rawInt.compareTo(maxWei) <= 0) {
                    inferredMappingSlot.put(addrKey, p);
                    return rawInt.doubleValue() / 1e18;
                }
            } catch (Exception ignored) {
            }
        }
        return 0.0;
    }

    public double queryTotalDeposited(String contractAddress, double vaultEthHint) {
        String addrKey = contractAddress.toLowerCase();
        List<String> sigs = new ArrayList<>(dynamicGlobalGetters.getOrDefault(addrKey, Collections.emptyList()));
        for (String std : STANDARD_GLOBAL_LIABILITY_SIGS) {
            if (!sigs.contains(std)) {
                sigs.add(std);
            }
        }
        for (String sig : sigs) {
            double val = callUint256Ether(contractAddress, sig);
            if (val > 0.0) {
                return val;
            }
        }
        return inferGlobalLiabilityFromStorage(contractAddress, vaultEthHint);
    }

    public double queryAgentStake(String contractAddress, String userAddr, String fallbackTokenAddress) {
        String uAddr = userAddr != null ? userAddr : agentAddress;
        String addrKey = contractAddress.toLowerCase();
        List<String> sigs = new ArrayList<>(dynamicUserGetters.getOrDefault(addrKey, Collections.emptyList()));
        for (String std : STANDARD_USER_STAKE_SIGS) {
            if (!sigs.contains(std)) {
                sigs.add(std);
            }
        }
        for (String sig : sigs) {
            double val = callUint256Ether(contractAddress, sig, uAddr);
            if (val > 0.0) {
                return val;
            }
        }
        double inferred = inferMappingBalanceFromStorage(contractAddress, uAddr);
        if (inferred > 0.0) {
            return inferred;
        }
        if (fallbackTokenAddress != null && !fallbackTokenAddress.isBlank()) {
            double tokVal = callUint256Ether(fallbackTokenAddress, "balanceOf(address)(uint256)", uAddr);
            if (tokVal > 0.0) {
                return tokVal;
            }
        }
        return 0.0;
    }

    public Map<String, String> verifyNotOwner(String contractAddress, String ownerSig) {
        List<String> candidates = List.of(
                ownerSig != null ? ownerSig : "owner()(address)",
                "owner()(address)",
                "admin()(address)",
                "governance()(address)"
        );
        String contractOwner = "0x0000000000000000000000000000000000000000";
        for (String sig : candidates) {
            String[] res = runCast(List.of("call", contractAddress, sig));
            if ("0".equals(res[2]) && res[0].startsWith("0x")) {
                contractOwner = res[0].split("\\s+")[0];
                break;
            }
        }
        if (contractOwner.equalsIgnoreCase(agentAddress)) {
            throw new SecurityException(
                    "Pelanggaran privilese: Agen " + agentAddress + " adalah owner dari " + contractAddress + "!"
            );
        }
        Map<String, String> info = new LinkedHashMap<>();
        info.put("agent_address", agentAddress);
        info.put("contract_address", contractAddress);
        info.put("contract_owner", contractOwner);
        info.put("standing", "REGULAR_USER_EOA");
        return info;
    }

    public Map<String, String> snapshotContractStorage(String contractAddress) {
        String addrKey = contractAddress.toLowerCase();
        Map<String, String> snap = new LinkedHashMap<>();
        for (int i = 0; i < numTrackedSlots; i++) {
            String slotKey = "0x" + Integer.toHexString(i);
            snap.put(slotKey, getStorageAt(contractAddress, slotKey));
            String mapKey = computeSolidityMappingSlot(agentAddress, i);
            snap.put(mapKey, getStorageAt(contractAddress, mapKey));
        }
        observedSloadSlots.computeIfAbsent(addrKey, k -> new HashSet<>());
        lastStorageSnapshot.put(addrKey, snap);
        return snap;
    }

    private static String normalizeSlot(String slotRaw) {
        try {
            BigInteger bi = new BigInteger(slotRaw.replace("0x", ""), 16);
            return "0x" + bi.toString(16);
        } catch (Exception e) {
            return slotRaw.toLowerCase();
        }
    }

    /**
     * Evaluasi hibrida jejak dataflow EVM untuk fitur x6 (SSTORE) dan x7 (SLOAD -> SSTORE waypoint).
     */
    public double[] traceTransactionDataflow(
            String txHash,
            String contractAddress,
            Map<String, String> preStorage
    ) {
        String addrKey = contractAddress.toLowerCase();
        Set<String> priorSloads = observedSloadSlots.computeIfAbsent(addrKey, k -> new HashSet<>());

        if (txHash != null && txHash.startsWith("0x") && txHash.length() == 66) {
            String params = String.format(
                    "[\"%s\",{\"disableMemory\":true,\"disableStorage\":false,\"disableStack\":false}]",
                    txHash
            );
            String rawTrace = jsonRpcRaw("debug_traceTransaction", params);
            if (rawTrace != null && rawTrace.contains("\"structLogs\"")) {
                Set<String> txSloads = new HashSet<>();
                Set<String> txSstores = new HashSet<>();
                Matcher m = SLOAD_SSTORE_REGEX.matcher(rawTrace);
                while (m.find()) {
                    String op = m.group(1);
                    String stackBody = m.group(2);
                    String lastHex = null;
                    Matcher hm = HEX_TOKEN_REGEX.matcher(stackBody);
                    while (hm.find()) {
                        lastHex = hm.group(1);
                    }
                    if (lastHex != null) {
                        String norm = normalizeSlot(lastHex);
                        if ("SLOAD".equals(op)) {
                            txSloads.add(norm);
                        } else if ("SSTORE".equals(op)) {
                            txSstores.add(norm);
                        }
                    }
                }
                double hasSstore = txSstores.isEmpty() ? 0.0 : 1.0;
                double hasWaypoint = 0.0;
                for (String s : txSstores) {
                    if (priorSloads.contains(s)) {
                        hasWaypoint = 1.0;
                        break;
                    }
                }
                priorSloads.addAll(txSloads);
                return new double[]{hasSstore, hasWaypoint};
            }
        }

        // Fallback Black-Box: Perbandingan slot storage publik sebelum dan sesudah transaksi
        Map<String, String> prevSnap = preStorage != null
                ? preStorage
                : lastStorageSnapshot.getOrDefault(addrKey, Collections.emptyMap());
        Map<String, String> currSnap = new LinkedHashMap<>();
        Set<String> modifiedSlots = new HashSet<>();
        Set<String> activeSlots = new HashSet<>();

        List<String> slotsToCheck = new ArrayList<>();
        for (int i = 0; i < numTrackedSlots; i++) {
            slotsToCheck.add("0x" + Integer.toHexString(i));
            slotsToCheck.add(computeSolidityMappingSlot(agentAddress, i));
        }

        for (String slotKey : slotsToCheck) {
            String val = getStorageAt(contractAddress, slotKey);
            currSnap.put(slotKey, val);
            String normK = normalizeSlot(slotKey);
            try {
                BigInteger bi = new BigInteger(val.replace("0x", ""), 16);
                if (!bi.equals(BigInteger.ZERO)) {
                    activeSlots.add(normK);
                }
            } catch (Exception ignored) {
            }
            if (prevSnap.containsKey(slotKey) && !prevSnap.get(slotKey).equalsIgnoreCase(val)) {
                modifiedSlots.add(normK);
            }
        }

        double hasSstore = modifiedSlots.isEmpty() ? 0.0 : 1.0;
        double hasWaypoint = 0.0;
        for (String s : modifiedSlots) {
            if (priorSloads.contains(s)) {
                hasWaypoint = 1.0;
                break;
            }
        }
        priorSloads.addAll(activeSlots);
        lastStorageSnapshot.put(addrKey, currSnap);
        return new double[]{hasSstore, hasWaypoint};
    }

    public String extractTxHash(String rawTxOutput) {
        if (rawTxOutput != null && !rawTxOutput.isBlank()) {
            for (String line : rawTxOutput.split("\\R")) {
                String stripped = line.trim();
                if (stripped.startsWith("transactionHash")) {
                    Matcher m = TX_HASH_REGEX.matcher(stripped);
                    if (m.find()) {
                        return m.group(0);
                    }
                }
            }
            Matcher mAny = TX_HASH_REGEX.matcher(rawTxOutput);
            if (mAny.find()) {
                return mAny.group(0);
            }
        }
        return null;
    }

    public TxExecutionResult sendEoaTransaction(
            String targetAddress,
            String functionSig,
            List<String> callArgs,
            double ethValue,
            String senderPrivateKey,
            Double gasPriceGwei
    ) {
        String pk = (senderPrivateKey != null && !senderPrivateKey.isBlank())
                ? senderPrivateKey
                : agentPrivateKey;
        List<String> castArgs = new ArrayList<>();
        castArgs.add("send");
        castArgs.add(targetAddress);

        if (functionSig != null && !functionSig.isBlank()) {
            castArgs.add(functionSig);
            if (callArgs != null && !callArgs.isEmpty()) {
                castArgs.addAll(callArgs);
            }
        }
        if (ethValue > 0.0) {
            BigInteger weiVal = BigInteger.valueOf(Math.round(ethValue * 1e9))
                    .multiply(BigInteger.valueOf(1_000_000_000L));
            castArgs.add("--value");
            castArgs.add(weiVal.toString());
        }
        if (gasPriceGwei != null && gasPriceGwei > 0.0) {
            long weiGas = Math.round(gasPriceGwei * 1e9);
            castArgs.add("--gas-price");
            castArgs.add(String.valueOf(weiGas));
        }
        castArgs.add("--private-key");
        castArgs.add(pk);

        String[] res = runCast(castArgs);
        if ("0".equals(res[2])) {
            return new TxExecutionResult(true, res[0]);
        }
        return new TxExecutionResult(false, res[1]);
    }
}
