"""
Konfigurasi Terpusat Eksperimen Skripsi.

Mendefinisikan parameter global, path berkas, anggaran modal gas B,
horizon putaran T, dan pengaturan algoritma untuk pengujian faktorial skripsi.
"""

from pathlib import Path
from typing import Dict, Any

# Root Directories
CURR_DIR = Path(__file__).resolve().parent
CMAB_ROOT = CURR_DIR.parent
REPO_ROOT = CMAB_ROOT.parent

RESULTS_DIR = CURR_DIR / "results"
TELEMETRY_DIR = RESULTS_DIR / "telemetry"
FIGURES_DIR = RESULTS_DIR / "figures"

TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
DATASET_DIR = CMAB_ROOT / "datasets" / "SC-Bench" / "categorized_dataset" / "Vault_and_Liquidity"
if not DATASET_DIR.exists():
    DATASET_DIR = REPO_ROOT / "SC-Bench" / "categorized_dataset" / "Vault_and_Liquidity"

MOCK_TOKEN_SOL = REPO_ROOT / "anvil_env" / "src" / "MockToken.sol"
VULNERABLE_VAULT_SOL = REPO_ROOT / "anvil_env" / "src" / "VulnerableVault.sol"

# Toolchain Binaries
SOLC_BIN = "/home/opeteer/.local/bin/solc"
CAST_BIN = "/home/opeteer/.foundry/bin/cast"
ANVIL_BIN = "/home/opeteer/.foundry/bin/anvil"
RPC_URL = "http://127.0.0.1:8545"

# Anvil Accounts
DEPLOYER_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
DEPLOYER_ADDR = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"

PEER_PK = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
PEER_ADDR = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"

AGENT_PK = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"
AGENT_ADDR = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"

# Context Dimension
CONTEXT_DIM = 10

# Kondisi 3: Variasi Anggaran Modal Gas (Knapsack Budget B)
# Berdasarkan horizon default T = 50 putaran
DEFAULT_HORIZON_T = 50

BUDGET_TIERS: Dict[str, Dict[str, Any]] = {
    "tight": {
        "tier_id": "BUDGET_TIGHT",
        "name": "Anggaran Ketat (Tight Budget - B_low)",
        "budget_eth": 0.05,
        "target_coverage_pct": "20% - 30% dari T",
        "description": "Hanya cukup untuk ~10-15 putaran jika memanggil fungsi boros; menguji kepekaan pengali dual lambda_t.",
    },
    "moderate": {
        "tier_id": "BUDGET_MODERATE",
        "name": "Anggaran Sedang (Moderate Budget - B_med)",
        "budget_eth": 0.15,
        "target_coverage_pct": "50% - 60% dari T",
        "description": "Cukup untuk mengeksplorasi setengah horizon; menguji keseimbangan antara reward dan gas penalty.",
    },
    "loose": {
        "tier_id": "BUDGET_LOOSE",
        "name": "Anggaran Longgar (Loose/Abundant Budget - B_high)",
        "budget_eth": 0.50,
        "target_coverage_pct": "~100% dari T",
        "description": "Mencukupi seluruh horizon; memverifikasi apakah suku knapsack melandai dan tidak menghambat reward.",
    },
}

# Algorithmic Hyperparameters
ALPHA = 0.55
GAMMA = 0.82
L2_LAMBDA = 1.0
INITIAL_LAMBDA = 1.0
MIN_CAPITAL_THRESHOLD_ETH = 0.005  # Ambang kebangkrutan on-chain

POLICY_NAMES = [
    "UniformRandom",
    "LinUCB",
    "D-LinUCB",
    "LinCBwK",
    "D-LinCBwK",
]

