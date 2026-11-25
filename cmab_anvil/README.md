# Dokumentasi Teknis Paket `cmab_anvil` (FuzzCMAB)

Paket `cmab_anvil` mengimplementasikan arsitektur **Discounted Linear Contextual Multi-Armed Bandits with Knapsacks (D-LinCBwK)** untuk pengujian keamanan (*fuzzing*) *smart contract* berbasis transaksi EOA tanpa hak istimewa di atas simulator lokal **Anvil (Foundry)**.

---

## 1. Arsitektur Modul

```text
cmab_anvil/
├── __init__.py                               # Ekspor publik seluruh kelas kebijakan, Arm, env, dan oracle
├── src/
│   ├── algorithms/                           # Implementasi kebijakan CMAB & evaluator regret
│   │   ├── base.py                           # BaseCMABPolicy (ABC) & ArmSelectionResult
│   │   ├── discounted_linucb_knapsack.py     # DiscountedLinCBwK (D-LinCBwK - Metode Usulan)
│   │   ├── linucb_knapsack.py                # StationaryLinCBwK (LinCBwK: gamma = 1.0)
│   │   ├── discounted_linucb.py              # DiscountedLinUCB (D-LinUCB: lambda_t = 0)
│   │   ├── linucb.py                         # DisjointLinUCB (LinUCB: gamma = 1.0, lambda_t = 0)
│   │   ├── uniform_random.py                 # UniformRandomPolicy (Baseline Forge Fuzz)
│   │   └── dynamic_regret.py                 # DynamicRegretEvaluator & compare_policies_dynamic_regret
│   ├── common/                               # Lapisan komunikasi RPC Anvil & abstraksi environment
│   │   ├── anvil_rpc.py                      # UnprivilegedAnvilClient (Zero-Privilege + Hybrid Opcode Trace)
│   │   └── base_env.py                       # Arm (@dataclass frozen), StepTransition, BaseUnprivilegedBanditEnv
│   └── Agen_Vault_Custody/                   # Pemodelan domain kontrak Vault Custody
│       ├── context.py                        # OnChainContextSnapshot & PublicContextExtractor (x_t in R^10)
│       ├── arms.py                           # build_vault_custody_arm_catalog (K = 8 objek Arm)
│       ├── reward.py                         # VaultCustodyRewardOracle & BanditRewardBreakdown
│       └── agent.py                          # VaultCustodyEnv & VaultCustodyCMABAgent
├── experiments_skripsi/                      # Suite eksperimen faktorial multi-kondisi & generator grafik
└── tests/                                    # Rangkaian unit test pytest & skrip CLI simulasi
```

---

## 2. Spesifikasi Ruang Konteks ($x_t \in \mathbb{R}^{10}$) — `context.py`

`PublicContextExtractor.to_vector(snap)` memetakan `OnChainContextSnapshot` menjadi vektor $x_t \in \mathbb{R}^{10}$ yang bebas-skala (*scale-invariant*):

| Indeks | Nama Fitur (`FEATURE_NAMES`) | Rumus Normalisasi | Rentang | Makna Domain |
| :-: | :--- | :--- | :---: | :--- |
| $x_0$ | `x0_bias` | $1.0$ | $\{1.0\}$ | Suku intersep regresi ridge linier. |
| $x_1$ | `x1_vault_liquidity_ratio` | $\frac{\mathrm{VaultETH}}{\mathrm{VaultETH} + \mathrm{AgentETH} + \epsilon}$ | $[0.0, 1.0]$ | Proporsi likuiditas kontrak relatif terhadap modal agen. |
| $x_2$ | `x2_claim_liability_ratio` | $\frac{\mathrm{TotalDeposited}}{\mathrm{VaultETH} + \mathrm{TotalDeposited} + \epsilon}$ | $[0.0, 1.0]$ | Rasio beban kewajiban klaim terhadap aset kontrak. |
| $x_3$ | `x3_solvency_gap_ratio` | $\mathrm{clip}\left(\frac{\mathrm{VaultETH} - \mathrm{TotalDeposited}}{\max(\mathrm{TotalDeposited}, 1.0)}, -1, 1\right)$ | $[-1.0, 1.0]$ | Surplus ($>0$) atau defisit/insolvensi ($<0$) cadangan kontrak. |
| $x_4$ | `x4_agent_stake_ratio` | $\mathrm{clip}\left(\frac{\mathrm{AgentVaultStake}}{\mathrm{StakeDenom}}, 0, 1\right)$ | $[0.0, 1.0]$ | Proporsi saldo/*stake* tercatat milik agen di dalam kontrak. |
| $x_5$ | `x5_deposit_eligibility` | $\min\left(1.0, \frac{\mathrm{AgentEOAETH}}{\mathrm{MinDepositReq}}\right)$ | $[0.0, 1.0]$ | Kelayakan modal seketika untuk mengeksekusi fungsi *payable*. |
| $x_6$ | `x6_sstore_execution_flag` | $\mathbb{I}(\mathrm{tx}_{t-1} \text{ mengeksekusi } \mathtt{SSTORE})$ | $\{0.0, 1.0\}$ | Indikator apakah aksi terakhir mengubah storage persisten EVM. |
| $x_7$ | `x7_dataflow_waypoint_flag` | $\mathbb{I}(\mathtt{SSTORE}_{t-1} \cap \bigcup_{k < t-1} \mathtt{SLOAD}_k \neq \emptyset)$ | $\{0.0, 1.0\}$ | Indikator terbukanya ketergantungan data lintas-transaksi (*waypoint*). |
| $x_8$ | `x8_recent_revert_rate` | $\frac{1}{W}\sum_{k=t-W}^{t-1} \mathbb{I}(\tau_k = 0), \; W=8$ | $[0.0, 1.0]$ | Rata-rata bergerak tingkat kegagalan (*revert*) 8 transaksi terakhir. |
| $x_9$ | `x9_vault_code_alive` | $\mathbb{I}(\lvert\mathrm{Bytecode}(\mathrm{Vault})\rvert > 0)$ | $\{0.0, 1.0\}$ | Indikator keutuhan *bytecode* kontrak (`0.0` jika `selfdestruct`). |

---

## 3. Spesifikasi Kelas `Arm` & Ruang Aksi ($\mathcal{A}$) — `base_env.py` & `arms.py`

Setiap lengan direpresentasikan oleh objek *immutable* `Arm` (`@dataclass(frozen=True)` di `src/common/base_env.py`):

```python
@dataclass(frozen=True)
class Arm:
    arm_id: int            # Indeks numerik lengan (0 .. K-1)
    name: str              # Nama deskriptif strategi transaksi
    target_contract: str   # Nama kontrak sasaran ("VulnerableVault" / "MockToken")
    function_sig: str      # Signature ABI Solidity (mis. "deposit()", "emergencyWithdraw()", "")
    eth_value: float       # Nilai ETH msg.value (v_a) yang dikirim bersama transaksi
    description: str       # Dokumentasi tujuan pengujian lengan
```

Pada `Agen_Vault_Custody` (`src/Agen_Vault_Custody/arms.py`), fungsi `build_vault_custody_arm_catalog()` mendefinisikan $K = 8$ objek `Arm`:
* `Arm 0` (`vault_deposit_small`): `deposit()` dengan `0.25 ETH`
* `Arm 1` (`vault_deposit_medium`): `deposit()` dengan `1.00 ETH`
* `Arm 2` (`vault_withdraw`): `withdraw()` dengan `0.00 ETH`
* `Arm 3` (`vault_fallback_transfer`): `receive()` fallback dengan `0.10 ETH`
* `Arm 4` (`vault_deposit_zero_probe`): `deposit()` boundary probe dengan `0.00 ETH`
* `Arm 5` (`vault_emergency_withdraw_probe`): `emergencyWithdraw()` exploit probe dengan `0.00 ETH`
* `Arm 6` (`token_mint_user`): `MockToken.mint(agent, 250e18)`
* `Arm 7` (`token_transfer_vault`): `MockToken.transfer(vault, 50e18)`

---

## 4. Struktur Imbalan Berjenjang (`VaultCustodyRewardOracle`) — `reward.py`

`VaultCustodyRewardOracle.compute_reward(prev_snap, next_snap, tx_succeeded, arm_id)` memetakan setiap transisi state ke dalam 5 partisi kejadian saling lepas $\text{supp}(r_t) = \{0.00, 0.20, 0.50, 0.70, 1.00\}$:

| Partisi | Syarat Kejadian Transisi State | Komponen Aktif | Nilai Imbalan ($r_t$) |
| :-: | :--- | :--- | :-: |
| **`E0`** | Transaksi gagal / `REVERT` di EVM ($\tau_t = 0$) | — | **`0.00`** |
| **`E1`** | Transaksi sukses tanpa perubahan storage ($\tau_t = 1 \wedge \Delta S_{\text{storage},t} = 0$) | $E_{\text{exec}}$ | **`0.20`** |
| **`E2`** | Mutasi storage persisten tanpa ekstraksi aset ($\Delta S_{\text{storage},t} \neq 0 \wedge \Delta \text{Aset}_{\text{eoa},t} \le \epsilon$) | $E_{\text{exec}} + E_{\text{state}}$ | **`0.50`** |
| **`E3`** | Mutasi storage disertai ekstraksi aset ke EOA ($\Delta S_{\text{storage},t} \neq 0 \wedge \Delta \text{Aset}_{\text{eoa},t} > \epsilon$) | $E_{\text{exec}} + E_{\text{state}} + E_{\text{econ}}$ | **`0.70`** |
| **`E4`** | **Pelanggaran Invarian Keamanan** (*Solvency Breach* atau *Contract Selfdestructed*) | Override Dominan $E_{\text{vuln}}$ | **`1.00`** |

---

## 5. Lima Kebijakan Uji & Reduksi Ablasi (`src/algorithms/`)

Seluruh kebijakan mewarisi `BaseCMABPolicy` dengan antarmuka seragam `select_arm(context)` dan `update(arm_id, context, reward, cost=...)`:

| Kebijakan | Kelas Implementasi | Faktor Diskonto ($\gamma$) | Pengali Dual Gas ($\lambda_t$) | Peran dalam Studi Ablasi |
| :--- | :--- | :---: | :---: | :--- |
| **`UniformRandom`** | `UniformRandomPolicy` | — | $0$ | Baseline *fuzzer* acak tanpa panduan konteks maupun biaya. |
| **`LinUCB`** | `DisjointLinUCB` | $1.00$ | $0$ | Baseline bandit kontekstual stasioner tanpa kendali anggaran. |
| **`D-LinUCB`** | `DiscountedLinUCB` | $0.82$ | $0$ | Ablasi non-stasioner tanpa kendali anggaran Knapsack. |
| **`LinCBwK`** | `StationaryLinCBwK` | $1.00$ | Dinamis (Pers. 3.13) | Ablasi Knapsack stasioner tanpa pelupaan eksponensial. |
| **`D-LinCBwK`** | `DiscountedLinCBwK` | $0.82$ | Dinamis (Pers. 3.13) | **Metode Usulan** (adaptif non-stasioner + kendali anggaran Knapsack). |
