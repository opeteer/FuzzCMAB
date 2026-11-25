# Lingkungan Simulator Anvil untuk FuzzCMAB (`anvil_env`)

Direktori `anvil_env/` menyediakan lingkungan eksekusi **Anvil (Foundry)** lokal beserta kontrak uji kustodian (`VulnerableVault.sol` dan `MockToken.sol`) serta skrip otomasi penyiapan kedudukan **Pengguna Tanpa Hak Istimewa (*Zero-Privilege EOA Standing*)** untuk agen **FuzzCMAB**.

---

## 1. Arsitektur Interaksi On-Chain

```text
+-------------------------------------------------------------------+
|                   ANVIL LOCAL EVM SIMULATOR                       |
|  - JSON-RPC Endpoint : http://127.0.0.1:8545 (Chain ID: 31337)    |
|  - Flag Tracing      : --steps-tracing (debug_traceTransaction)   |
|  - Account #0 (Owner): 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266 |
|  - Account #1 (Peer) : 0x70997970C51812dc3A010C7d01b50e0d17dc79C8 |
|  - Account #2 (Agent): 0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC |
+-------------------------------------------------------------------+
                                 ▲
                                 │ 1. Read-Only Queries & Opcode Trace
                                 │    (eth_getBalance, eth_getStorageAt,
                                 │     eth_call, debug_traceTransaction)
                                 │ 2. Signed Unprivileged EOA Transactions
                                 │    (cast send --private-key AGENT_PK)
+--------------------------------┴----------------------------------+
|                     FUZZCMAB PYTHON ENGINE                        |
|  - PublicContextExtractor : Snapshot EVM -> x_t in R^10           |
|  - D-LinCBwK Policy       : argmax_a [r_hat(x_t) - lambda_t*c_hat]|
|  - VaultCustodyRewardOracle: Evaluasi Transisi E0..E4 in [0, 1]   |
+-------------------------------------------------------------------+
```

---

## 2. Struktur Direktori

```text
anvil_env/
├── README.md                        # Dokumentasi lingkungan simulator Anvil
├── foundry.toml                     # Konfigurasi kompilator Solidity Foundry
├── deployed_addresses.json          # Alamat kontrak hasil deployment aktif di Anvil
├── src/
│   ├── VulnerableVault.sol          # Kontrak tolok ukur Vault Custody (solvency & emergencyWithdraw bug)
│   └── MockToken.sol                # Kontrak token ERC-20 pendamping
├── script/
│   └── Deploy.s.sol                 # Skrip deployment Foundry (Account #0 sebagai Owner)
├── test/
│   └── VaultTest.t.sol              # Pengujian unit Foundry untuk VulnerableVault
├── scripts/
│   ├── start_anvil.sh               # Menjalankan proses Anvil standar di port 8545
│   ├── start_anvil_fair.sh          # Menjalankan Anvil dengan parameter eksperimen terkontrol
│   ├── stop_anvil.sh                # Menghentikan proses Anvil yang sedang berjalan
│   ├── deploy_contracts.sh          # Men-deploy VulnerableVault & MockToken via forge script
│   ├── setup_fair_agent.sh          # Memverifikasi bahwa Agen BUKAN owner & mengekspor artefak ABI
│   ├── export_artifacts.py          # Mengekstrak ABI & bytecode hasil kompilasi Foundry
│   ├── run_fair_simulation.sh       # Pipeline 1-klik: start_anvil_fair.sh + setup_fair_agent.sh
│   └── run_anvil_simulation.sh      # Pipeline 1-klik: start_anvil.sh + deploy + export_artifacts
└── fuzz_artifacts/                  # Artefak ABI & BIN hasil ekstraksi
```

---

## 3. Cara Penggunaan

### A. Menyiapkan Lingkungan Anvil Mode Adil (*Strict Fair Standing*)
```bash
bash anvil_env/scripts/run_fair_simulation.sh
```
Skrip ini akan:
1. Menjalankan node Anvil di `http://127.0.0.1:8545`.
2. Men-deploy `VulnerableVault` (didanai 10 ETH awal oleh Account #0 / Owner) dan `MockToken`.
3. Memverifikasi secara on-chain bahwa alamat EOA agen bukan `owner()` dari `VulnerableVault`.
4. Menyimpan alamat kontrak ke `anvil_env/deployed_addresses.json`.

### B. Menghentikan Node Anvil
```bash
bash anvil_env/scripts/stop_anvil.sh
```
