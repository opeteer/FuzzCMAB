# FuzzCMAB Java Simulation Suite (`cmab_anvil_java`)

Folder ini berisi **implementasi lengkap bahasa Java (Java 17+ / Maven)** dari keseluruhan sistem simulasi **FuzzCMAB** (*Discounted Linear Contextual Multi-Armed Bandits with Knapsacks* / **D-LinCBwK** pada simulator **Anvil EVM**), yang disalin secara utuh dari paket Python [`cmab_anvil/`](file:///home/opeteer/FuzzMaster/cmab_anvil/) tanpa menghapus atau mengurangi simulasi Python yang sudah ada.

---

## 1. Pemetaan Modul Python $\leftrightarrow$ Kelas Java

| Modul Python (`cmab_anvil/`) | Kelas Java (`cmab_anvil_java/src/main/java/id/ac/skripsi/fuzzcmab/`) | Fungsi & Peran |
| :--- | :--- | :--- |
| `src/algorithms/base.py` | `algorithms/BaseCMABPolicy.java` & `algorithms/ArmSelectionResult.java` | Abstraksi dasar kebijakan CMAB & hasil seleksi lengan. |
| `src/algorithms/discounted_linucb_knapsack.py` | `algorithms/DiscountedLinCBwK.java` | **Metode Usulan D-LinCBwK** (Persamaan 3.13–3.21 & Algoritma 3.1). |
| `src/algorithms/linucb_knapsack.py` | `algorithms/StationaryLinCBwK.java` | Ablasi Knapsack stasioner (`LinCBwK`, $\gamma = 1.0$). |
| `src/algorithms/discounted_linucb.py` | `algorithms/DiscountedLinUCB.java` | Ablasi non-stasioner tanpa Knapsack (`D-LinUCB`, $\lambda_t = 0$). |
| `src/algorithms/linucb.py` | `algorithms/DisjointLinUCB.java` | Baseline bandit linier stasioner (`LinUCB`, $\gamma = 1.0, \lambda_t = 0$). |
| `src/algorithms/uniform_random.py` | `algorithms/UniformRandomPolicy.java` | Baseline pemilih acak seragam (`UniformRandom`). |
| `src/algorithms/dynamic_regret.py` | `algorithms/DynamicRegretEvaluator.java` | Evaluator *Cumulative Dynamic Regret* $\text{D-Regret}(T)$ (Persamaan 3.22). |
| `src/common/base_env.py` | `common/Arm.java`, `common/StepTransition.java`, `common/BaseRegularUserBanditEnv.java` | Struktur data *immutable* `Arm`, `StepTransition`, dan kelas induk environment agen pengguna biasa. |
| `src/common/anvil_rpc.py` | `common/RegularUserAnvilClient.java` | Klien JSON-RPC & Foundry `cast` agen pengguna biasa, hash Ethereum Keccak-256 murni, resolusi ABI/storage 4-Tier, & *opcode trace* `SLOAD`/`SSTORE`. |
| `src/Agen_Vault_Custody/context.py` | `vaultcustody/OnChainContextSnapshot.java` & `vaultcustody/PublicContextExtractor.java` | Ekstraksi vektor konteks 10 dimensi $x_t \in \mathbb{R}^{10}$ (Tabel 3.1). |
| `src/Agen_Vault_Custody/arms.py` | `vaultcustody/VaultCustodyArms.java` | Katalog $K = 8$ objek `Arm` untuk `VulnerableVault` & `MockToken`. |
| `src/Agen_Vault_Custody/reward.py` | `vaultcustody/BanditRewardBreakdown.java` & `vaultcustody/VaultCustodyRewardOracle.java` | Orakel imbalan berjenjang 4-komponen $\text{supp}(r_t) = \{0.00, 0.20, 0.50, 0.70, 1.00\}$. |
| `src/Agen_Vault_Custody/agent.py` | `vaultcustody/VaultCustodyEnv.java` & `vaultcustody/VaultCustodyCMABAgent.java` | Environment `VulnerableVault` (rezim gas + transaksi pihak luar) & orkestrator agen. |
| `experiments_skripsi/config.py` | `experiments/ExperimentConfig.java` | Konfigurasi terpusat anggaran `BUDGET_TIERS`, $\alpha=0.55$, $\gamma=0.82$, dan akun Anvil. |
| `experiments_skripsi/gas_regimes.py` | `experiments/GasRegimes.java` | Generator deret harga gas Rezim 1 (`stable`) & Rezim 2 (`volatile_spikes`). |
| `experiments_skripsi/contract_catalog.py` | `experiments/ContractCatalog.java` | Katalog 8 kontrak (`SKRIPSI-B01` s.d. `SKRIPSI-D03`). |
| `tests/run_cmab_simulation.py` | `experiments/SimulationRunner.java` | CLI runner simulasi tunggal `Agen_Vault_Custody`. |
| `experiments_skripsi/runner.py` | `experiments/SkripsiExperimentRunner.java` | Runner utama matriks eksperimen faktorial multi-kondisi. |
| `tests/test_*.py` | `src/test/java/id/ac/skripsi/fuzzcmab/FuzzCMABTestSuite.java` | Suite pengujian unit matematis & fungsional Java. |

---

## 2. Cara Kompilasi & Menjalankan Simulasi Java

Proyek ini dirancang **100% mandiri (*zero external dependency*)** menggunakan pustaka standar Java 17 (`java.base` dan `java.net.http`), sehingga dapat dikompilasi dan dijalankan langsung menggunakan skrip [`build_and_run.sh`](file:///home/opeteer/FuzzMaster/cmab_anvil_java/build_and_run.sh) maupun menggunakan Apache Maven (`pom.xml`):

### A. Menggunakan Skrip `build_and_run.sh` (Tanpa Maven)
```bash
chmod +x cmab_anvil_java/build_and_run.sh

# 1. Menjalankan Unit Test Suite Java
./cmab_anvil_java/build_and_run.sh test

# 2. Menjalankan Simulasi Tunggal Agen_Vault_Custody dengan D-LinCBwK
./cmab_anvil_java/build_and_run.sh sim --algorithm d-lincbwk --gas-regime volatile_spikes --budget 2.00 --rounds 15

# 3. Menjalankan Matriks Eksperimen Faktorial Skripsi
./cmab_anvil_java/build_and_run.sh matrix --tier benchmark --regime all --budget all --horizon 50 --seeds 1
```

### B. Menggunakan Apache Maven (`pom.xml`)
```bash
cd cmab_anvil_java
mvn compile
mvn exec:java -Dexec.mainClass="id.ac.skripsi.fuzzcmab.experiments.SkripsiExperimentRunner" -Dexec.args="--tier shallow --regime volatile_spikes --budget moderate --horizon 30"
```
