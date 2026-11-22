# FuzzCMAB: Contextual Multi-Armed Bandit Smart Contract Fuzzing & Exploration Framework

Framework penelitian berbasis **Contextual Multi-Armed Bandits (CMAB)** untuk eksplorasi, fuzzing interaksi, dan deteksi anomali pada *smart contract* berbasis EVM di atas lingkungan simulator lokal **Anvil (Foundry)**.

Repositori ini mereformulasi permasalahan interaksi kontrak dari agen *unprivileged user* (pengguna biasa tanpa hak akses `owner`/admin) menggunakan algoritma **Disjoint LinUCB** (stasioner) dan **Discounted LinUCB / D-LinUCB** (adaptif terhadap lingkungan non-stasioner).

---

## 📁 Struktur Proyek

```text
.
├── cmab_anvil/                  # Framework inti riset CMAB
│   ├── src/                     # [Sumber Kode] Modul algoritma CMAB, agen, dan RPC
│   │   ├── algorithms/          # BaseCMABPolicy, LinUCB, Discounted LinUCB
│   │   ├── Agen_Vault_Custody/  # Agen spesialis Vault Custody (context, arms, reward, agent)
│   │   └── common/              # Unprivileged Anvil RPC client & Base CMAB Environment
│   ├── tests/                   # [File Simulasi] Skrip eksekusi simulasi & benchmark
│   │   ├── run_scbench_10contracts.py # Benchmark SC-Bench (Dual Scenario: Fixed & Capital)
│   │   ├── run_cmab_simulation.py     # Simulasi interaksi CMAB tunggal
│   │   └── run_comparative_experiment.py # Eksperimen komparasi LinUCB vs D-LinUCB
│   ├── datasets/                # [Datasets] Seluruh dataset smart contract & alat kurasi
│   │   ├── SC-Bench/            # Dataset kurasi smart contract SC-Bench
│   │   │   ├── categorized_dataset/   # Kategori Vault_and_Liquidity, AMM, Lending, dll.
│   │   │   └── dataset/               # Dataset asli SC-Bench
│   │   ├── categorize_scbench.py      # Script otomatisasi klasifikasi kontrak
│   │   └── scbench_summary_report.json # Statistik distribusi kategori kontrak
│   └── results/                 # [Hasil] Telemetri, checkpoint, laporan, & visualisasi
│       ├── charts/              # Grafik visualisasi PNG hasil evaluasi
│       ├── *.json               # Data telemetri hasil eksperimen & checkpoint model
│       ├── report.md            # Laporan utama evaluasi empiris 11 kontrak SC-Bench
│       └── laporan_worst_gas_no_regime_double_capital.md # Analisis ketahanan modal
└── anvil_env/                   # Lingkungan lokal Foundry & Anvil
    ├── src/                     # Kontrak uji lokal (MockToken.sol, VulnerableVault.sol)
    ├── scripts/                 # Otomasi deployment Anvil & setup agen tanpa privilese
    └── deployed_addresses.json  # Alamat kontrak hasil deploy lokal
```

---

## 🚀 Panduan Memulai Cepat

### 1. Prasyarat Sistem
- **Python ≥ 3.10** dengan pustaka: `numpy`, `matplotlib`
- **Foundry / Anvil**: Terinstal di path sistem (`anvil`, `cast`, `forge`)

### 2. Memastikan Node Anvil Berjalan
Node Anvil dapat dijalankan di latar belakang pada port default `8545`:
```bash
./anvil_env/scripts/start_anvil.sh
```
Atau menggunakan Anvil CLI secara langsung:
```bash
anvil --port 8545 --chain-id 31337 --accounts 10 --balance 10000 --gas-limit 30000000
```

### 3. Menjalankan Benchmark Evaluasi SC-Bench
Eksekusi evaluasi komparasi LinUCB vs D-LinUCB pada smart contract SC-Bench:
```bash
# Skenario Horizon Tetap (Fixed Horizon, T = 24 rounds)
python3 cmab_anvil/tests/run_scbench_10contracts.py --scenario fixed --gas-mode static

# Skenario Ketahanan Modal & Gas Fluktuatif (Capital Exhaustion Scenario)
python3 cmab_anvil/tests/run_scbench_10contracts.py --scenario capital --gas-mode worst_case --capital 0.50
```

### 4. Menjalankan Simulasi Tunggal (Agen Vault Custody)
```bash
# Simulasi menggunakan Discounted LinUCB
python3 cmab_anvil/tests/run_cmab_simulation.py --algorithm d-linucb --gamma 0.85 --rounds 10

# Simulasi menggunakan Disjoint LinUCB
python3 cmab_anvil/tests/run_cmab_simulation.py --algorithm linucb --alpha 1.25 --rounds 8
```

---

## 📊 Hasil Evaluasi & Dokumentasi
- Laporan lengkap perancangan metodologi & hasil empiris dapat dilihat di [`cmab_anvil/results/report.md`](cmab_anvil/results/report.md).
- Analisis ketahanan modal pada kondisi worst-case gas dapat dilihat di [`cmab_anvil/results/laporan_worst_gas_no_regime_double_capital.md`](cmab_anvil/results/laporan_worst_gas_no_regime_double_capital.md).
- Grafik perbandingan trajektori dan performa tersedia di folder [`cmab_anvil/results/charts/`](cmab_anvil/results/charts/).

---

## 📜 Lisensi
Proyek ini dilisensikan di bawah ketentuan lisensi yang tertera pada berkas [LICENSE](LICENSE).
