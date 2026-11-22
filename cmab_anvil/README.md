# Pemodelan Contextual Multi-Armed Bandits (CMAB) pada Simulator Anvil

Folder ini (`cmab_anvil/`) berisi implementasi **FuzzCMAB**: kerangka kerja **Contextual Multi-Armed Bandits (CMAB)** untuk eksplorasi & interaksi *smart contract* oleh **pengguna cerdas tanpa hak istimewa (*unprivileged intelligent user*)** di atas jaringan lokal **Anvil (Foundry)**.

---

## 📁 Arsitektur & Struktur Direktori

Struktur modular `cmab_anvil` ditata rapi ke dalam 4 folder utama:

```text
cmab_anvil/
├── __init__.py                         # Package init & re-export backward compatibility
├── README.md                           # Dokumentasi teknis modul cmab_anvil
├── src/                                # [SUMBER KODE] Seluruh logika implementasi algoritma & agen
│   ├── __init__.py
│   ├── algorithms/                     # Kebijakan & strategi CMAB
│   │   ├── base.py                     # BaseCMABPolicy, ArmSelectionResult
│   │   ├── linucb.py                   # LinUCBTemplate, DisjointLinUCB, HybridLinUCBTemplate
│   │   └── discounted_linucb.py        # DiscountedLinUCBTemplate, DiscountedLinUCB
│   ├── common/                         # Komponen universal RPC & Environment
│   │   ├── anvil_rpc.py                # UnprivilegedAnvilClient
│   │   └── base_env.py                 # BaseUnprivilegedBanditEnv, StepTransition, BaseArmSpec
│   └── Agen_Vault_Custody/             # Domain agen spesialis Vault & Custody
│       ├── context.py                  # Ekstraktor konteks on-chain publik x_t in R^{10}
│       ├── arms.py                     # Katalog 8 lengan aksi transaksi pengguna biasa
│       ├── reward.py                   # Oracle reward deteksi anomali/keberhasilan
│       └── agent.py                    # VaultCustodyEnv & VaultCustodyCMABAgent
├── tests/                              # [SIMULASI & BENCHMARK] Skrip simulasi dan pengujian
│   ├── __init__.py
│   ├── run_cmab_simulation.py          # Simulasi interaksi CMAB tunggal (Agen Vault Custody)
│   ├── run_comparative_experiment.py   # Eksperimen komparasi LinUCB vs D-LinUCB
│   └── run_scbench_10contracts.py      # Benchmark evaluasi 11 kontrak SC-Bench (Dual Scenario)
├── datasets/                           # [DATASET] Dataset smart contract & alat kurasi
│   ├── SC-Bench/                       # Dataset kurasi smart contract SC-Bench
│   │   ├── categorized_dataset/        # Kontrak terkategori (Vault_and_Liquidity, AMM, dll.)
│   │   └── dataset/                    # Dataset mentah SC-Bench
│   ├── categorize_scbench.py           # Skrip otomatisasi klasifikasi kontrak SC-Bench
│   └── scbench_summary_report.json     # Laporan statistik distribusi kategori kontrak
└── results/                            # [HASIL & LAPORAN] Telemetri, checkpoint, grafik, & laporan
    ├── charts/                         # Grafik visualisasi PNG hasil evaluasi
    ├── *.json                          # Telemetri hasil eksperimen & checkpoint kebijakan
    ├── report.md                       # Laporan evaluasi non-stasioner LinUCB vs D-LinUCB
    └── laporan_worst_gas_no_regime_double_capital.md # Laporan ketahanan modal worst-case gas
```

---

## 🚀 Cara Menjalankan Simulasi & Pengujian

### 1. Memastikan Node Anvil Aktif
```bash
./anvil_env/scripts/start_anvil.sh
```

### 2. Menjalankan Simulasi Tunggal (Agen Vault Custody)
```bash
# Discounted LinUCB (D-LinUCB)
python3 cmab_anvil/tests/run_cmab_simulation.py --algorithm d-linucb --gamma 0.85 --rounds 10

# Disjoint LinUCB Stasioner
python3 cmab_anvil/tests/run_cmab_simulation.py --algorithm linucb --alpha 1.25 --rounds 8
```

### 3. Menjalankan Pengujian Komparasi Head-to-Head
```bash
python3 cmab_anvil/tests/run_comparative_experiment.py
```

### 4. Menjalankan Benchmark Skala Penuh pada Dataset SC-Bench
```bash
# Skenario A (Fixed Horizon, T = 24 Rounds)
python3 cmab_anvil/tests/run_scbench_10contracts.py --scenario fixed --gas-mode static

# Skenario B (Capital Exhaustion, Worst-Case Gas & Stochastic Events)
python3 cmab_anvil/tests/run_scbench_10contracts.py --scenario capital --gas-mode worst_case --capital 0.50
```
