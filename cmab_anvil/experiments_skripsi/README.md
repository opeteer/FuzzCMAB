# Panduan Eksperimen Faktorial Skripsi (`experiments_skripsi`)

Paket `cmab_anvil/experiments_skripsi` mengeksekusi matriks pengujian faktorial multi-kondisi untuk mengevaluasi kinerja **D-LinCBwK** dibandingkan empat metode pembanding (`UniformRandom`, `LinUCB`, `D-LinUCB`, `LinCBwK`) di simulator **Anvil EVM**.

---

## 1. Desain Matriks Eksperimen Faktorial

### Kondisi 1: Kelompok & Kedalaman State Kontrak Target (`--tier`)
Didefinisikan di [`contract_catalog.py`](file:///home/opeteer/FuzzMaster/cmab_anvil/experiments_skripsi/contract_catalog.py):
1. **`benchmark` (Kontrak Sintetis Terkontrol):**
   * `SKRIPSI-B01` (`VulnerableVault.sol`): Kontrak kustodian ETH dengan prasyarat likuiditas $\ge 5.00\text{ ETH}$ pada celah `emergencyWithdraw()`.
2. **`shallow` (Kedalaman State Rendah — SC-Bench):**
   * `SKRIPSI-S01` (`ClashForStash`): Kustodian token dengan celah *uninitialized `trustedSigner` (`address(0)`) ECDSA bypass*.
   * `SKRIPSI-S02` (`Token`): Token likuiditas berpajak dengan fungsi publik `withdraw()` tanpa proteksi kepemilikan.
3. **`medium` (Kedalaman State Sedang — SC-Bench):**
   * `SKRIPSI-M01` (`DrawTT`): Kontrak *PullPayment escrow* & NFT dengan biaya cetak dinamis.
   * `SKRIPSI-M02` (`Shuffle`): Kontrak *bidding pool* dengan celah pembajakan `fundsWallet` melalui `modifyRoom()`.
4. **`deep` (Kedalaman Multi-State Tinggi — SC-Bench):**
   * `SKRIPSI-D01` (`PyramidToken`): Kontrak *staking* terkunci 30 hari dengan bonus referal tak berjaminan (*unbacked referral inflation*).
   * `SKRIPSI-D02` (`PymeStaking`): Kontrak *staking* multi-tier dengan pengurasan cadangan bunga tinggi (`claim`).
   * `SKRIPSI-D03` (`Airdrop`): Kontrak distribusi kustodian berbasis verifikasi *Merkle proof*.

### Kondisi 2: Rezim Dinamika Harga Gas Jaringan (`--regime`)
Didefinisikan di [`gas_regimes.py`](file:///home/opeteer/FuzzMaster/cmab_anvil/experiments_skripsi/gas_regimes.py) dan disinkronkan menggunakan *Common Random Numbers* (CRN):
* **`stable` (Rezim 1 — Jaringan Stabil):** Distribusi Log-Normal EIP-1559 ($\mu = \ln(15), \sigma = 0.35$) pada rentang $[8.0, 32.0]\text{ Gwei}$.
* **`volatile_spikes` (Rezim 2 — Fluktuatif & Lonjakan Ekstrem):** Basis Uniform $[60, 140]\text{ Gwei}$ ditambah peluang $25\%$ lonjakan Pareto ($\alpha = 1.8 \times 80$) hingga $[60.0, 500.0]\text{ Gwei}$.

### Kondisi 3: Variasi Anggaran Modal Gas $B$ (`--budget`)
Didefinisikan di [`config.py`](file:///home/opeteer/FuzzMaster/cmab_anvil/experiments_skripsi/config.py):
* **`tight` ($B_{\text{low}} = 0.05\text{ ETH}$):** Menguji sensitivitas pengali dual $\lambda_t$ pada modal sangat terbatas.
* **`moderate` ($B_{\text{med}} = 0.15\text{ ETH}$):** Menguji keseimbangan eksplorasi-eksploitasi di bawah kendala biaya moderat.
* **`loose` ($B_{\text{high}} = 0.50\text{ ETH}$):** Menguji konvergensi ketika anggaran melimpah.

---

## 2. Lima Metrik Evaluasi & Uji Statistik

Pada setiap sel eksperimen, [`runner.py`](file:///home/opeteer/FuzzMaster/cmab_anvil/experiments_skripsi/runner.py) menghitung:
1. **Cumulative Dynamic Regret ($\text{D-Regret}(T)$):** Selisih kumulatif terhadap orakel dinamis (termasuk penalti penuh $r_t^*$ pada sisa putaran $\{\tau+1, \dots, T\}$ jika agen bangkrut dini di putaran $\tau < T$).
2. **Frekuensi Pemicuan Kerentanan (`E4`), Mutasi Storage (`SSTORE`), *Dataflow Waypoint*, & Imbalan Kumulatif.**
3. **Rasio Efisiensi Penggunaan Anggaran Gas ($\text{Eff}_{\text{gas}}$):**
   $$\text{Eff}_{\text{gas}} = \frac{\sum_{t=1}^{\tau} r_t(a_t)}{\sum_{t=1}^{\tau} c_t(a_t)}$$
4. **Tingkat Kelangsungan Hidup Anggaran (*Budget Survival Rate*):** $\text{SR} = \frac{\tau}{T} \times 100\%$.
5. **Overhead Komputasi Keputusan (ms/putaran) & Uji Signifikansi Statistik:** Interval kepercayaan $95\%$ (Student-$t$), *Paired two-sided t-test*, dan *Wilcoxon Signed-Rank Test* antar-replikasi (`--seeds M`).

---

## 3. Perintah Eksekusi

```bash
# Menjalankan kontrak benchmark (VulnerableVault) di semua rezim & anggaran
python3 cmab_anvil/experiments_skripsi/runner.py --tier benchmark --regime all --budget all --horizon 50 --seeds 1

# Menjalankan tier tertentu (mis. shallow / medium / deep)
python3 cmab_anvil/experiments_skripsi/runner.py --tier shallow --regime all --budget moderate --horizon 30 --seeds 1

# Menjalankan matriks penuh dengan M = 5 replikasi seed CRN
python3 cmab_anvil/experiments_skripsi/runner.py --tier all --regime all --budget all --horizon 50 --seeds 5
```

---

## 4. Pembuatan Grafik Publikasi (300 DPI PNG & PDF)

```bash
python3 cmab_anvil/experiments_skripsi/generate_plots.py
```

Menghasilkan 4 grafik di `cmab_anvil/experiments_skripsi/results/figures/`:
1. `fig_dynamic_regret_comparison.png` / `.pdf`: Trajektori *Cumulative Dynamic Regret* $\text{D-Regret}(t)$ kelima kebijakan.
2. `fig_capital_survival_tight_budget.png` / `.pdf`: Trajektori penyusutan anggaran $B_t$ dan *survival rate*.
3. `fig_gas_spike_adaptation.png` / `.pdf`: Respons harga bayangan dual $\lambda_t$ terhadap lonjakan harga gas (`volatile_spikes`).
4. `fig_gas_efficiency_and_vuln.png` / `.pdf`: Perbandingan rata-rata $\text{Eff}_{\text{gas}}$ dan total penemuan kerentanan `E4`.
