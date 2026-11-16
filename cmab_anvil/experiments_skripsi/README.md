# Panduan Eksperimen Skripsi: Evaluasi Ketahanan & Efisiensi D-LinCBwK pada Smart Contract EVM

Folder ini dirancang khusus untuk pengujian faktorial skripsi secara terisolasi, terstruktur, dan dapat direproduksi (*self-contained & fully reproducible*).

---

## 1. Tiga Kondisi Eksperimen

1. **Kondisi 1: Variasi Kompleksitas Kedalaman State Kontrak Target**
   - **`shallow`**: `ClashForStash` & `Token` (1–2 mutasi storage linier per aksi).
   - **`medium`**: `DrawTT` & `Shuffle` (status berurutan, escrow settlement asinkron).
   - **`deep`**: `PyramidToken`, `PymeStaking`, & `Airdrop` (dependensi waktu, staking maturity, dan Merkle tree).
2. **Kondisi 2: Skenario Dinamika Harga Gas di Anvil**
   - **`stable` (Rezim 1)**: Fluktuasi sempit $15 - 25\text{ Gwei}$.
   - **`volatile_spikes` (Rezim 2)**: Gelombang dasar $20 - 50\text{ Gwei}$ dengan lonjakan mendadak (*traffic spike*) mencapai $100 - 200\text{ Gwei}$ (peluang 25%).
3. **Kondisi 3: Variasi Anggaran Modal Gas ($B$)**
   - **`tight` ($B_{\text{low}} = 0.05\text{ ETH}$)**: Hanya cukup untuk $\approx 20\% - 30\%$ horizon $T$.
   - **`moderate` ($B_{\text{med}} = 0.15\text{ ETH}$)**: Cukup untuk $\approx 50\% - 60\%$ horizon $T$.
   - **`loose` ($B_{\text{high}} = 0.50\text{ ETH}$)**: Cukup untuk seluruh horizon $T$.

---

## 2. Lima Kebijakan yang Dibandingkan

1. **`UniformRandom`**: Baseline acak seragam (representasi Forge Fuzz / AFL).
2. **`LinUCB`**: Bandit linier stasioner tanpa kendala gas ($\gamma = 1.0$).
3. **`D-LinUCB`**: Bandit linier diskon non-stasioner tanpa kendala gas ($\gamma = 0.82$).
4. **`LinCBwK`**: Bandit knapsack stasioner dengan pengali dual gas $\lambda_t$ ($\gamma = 1.0$).
5. **`D-LinCBwK`**: **Metode usulan** menggabungkan diskon non-stasioner ($\gamma = 0.82$) dan pengali dual gas $\lambda_t$.

---

## 3. Cara Menjalankan Eksperimen

Semua eksperimen dijalankan melalui `runner.py`:

```bash
# 1. Quick Test (1 Kontrak Shallow, 10 Putaran, Rezim Volatile, Anggaran Ketat)
python3 cmab_anvil/experiments_skripsi/runner.py --tier shallow --regime volatile_spikes --budget tight --horizon 10 --contracts-limit 1

# 2. Menjalankan Satu Kondisi Spesifik (Misal: Semua Kontrak Deep pada Rezim Gas Spikes)
python3 cmab_anvil/experiments_skripsi/runner.py --tier deep --regime volatile_spikes --budget moderate --horizon 24

# 3. Menjalankan Matriks Penuh (Faktorial Penuh 3 Tiers x 2 Regimes x 3 Budgets x 5 Kebijakan)
python3 cmab_anvil/experiments_skripsi/runner.py --tier all --regime all --budget all --horizon 24
```

---

## 4. Visualisasi & Pembuatan Grafik Publikasi (300 DPI)

Setelah simulasi selesai, hasil telemetri otomatis tersimpan di `results/telemetry/`. Jalankan perintah berikut untuk menghasilkan grafik publikasi (format PNG dan PDF vektor):

```bash
python3 cmab_anvil/experiments_skripsi/generate_plots.py
```

Grafik akan tersimpan di `results/figures/`:
- `fig_capital_survival_tight_budget.png` / `.pdf`: Trajektori ketahanan modal pada anggaran ketat ($B_{\text{low}}$).
- `fig_gas_spike_adaptation.png` / `.pdf`: Respons adaptasi pengali $\lambda_t$ saat menghadapi lonjakan gas mendadak di Rezim 2.

