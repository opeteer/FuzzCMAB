#!/usr/bin/env python3
"""
Automated Comparative Evaluation Runner:
Executes both Disjoint LinUCB and Discounted LinUCB (D-LinUCB) on Anvil
under identical starting conditions, evaluates the empirical results,
and outputs comparative telemetry for reporting.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

curr = Path(__file__).resolve()
REPO_ROOT = curr.parents[2] if len(curr.parents) > 2 else curr.parents[-1]
for p in curr.parents:
    if (p / "anvil_env").exists():
        REPO_ROOT = p
        break

CMAB_ROOT = REPO_ROOT / "cmab_anvil"
RESULTS_DIR = CMAB_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CMAB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CMAB_ROOT / "src"))

SETUP_SCRIPT = REPO_ROOT / "anvil_env" / "scripts" / "setup_fair_agent.sh"
RPC_URL = "http://127.0.0.1:8545"

LINUCB_OUT = RESULTS_DIR / "linucb_simulation_results.json"
DLINUCB_OUT = RESULTS_DIR / "d_linucb_simulation_results.json"
SIM_SCRIPT = CMAB_ROOT / "tests" / "run_cmab_simulation.py"


def run_cmd(cmd: str, desc: str) -> None:
    print(f"\n[*] {desc}...")
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(f"[!] Error executing: {cmd}\nOutput:\n{p.stdout}")
        sys.exit(1)
    print(p.stdout)


def main():
    print("=" * 88)
    print("🔬 EKSPERIMEN KOMPARATIF CMAB ANVIL: LINUCB VS DISCOUNTED LINUCB (D-LINUCB)")
    print("   Dataset SC-Bench: Kategori Vault_and_Liquidity (K = 8 Arms)")
    print("=" * 88)

    # -------------------------------------------------------------------------
    # 1. Eksperimen 1: Disjoint LinUCB (Stasioner, gamma = 1.0)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("🔵 [FASE 1] PENYIAPAN & EKSEKUSI EKSPERIMEN 1: DISJOINT LINUCB")
    print("=" * 88)
    run_cmd(f'bash "{SETUP_SCRIPT}" "{RPC_URL}"', "Deploy Kontrak Bersih untuk LinUCB")

    run_cmd(
        f'python3 "{SIM_SCRIPT}" '
        f'--algorithm linucb --alpha 0.50 --rounds 10 --output-log "{LINUCB_OUT}"',
        "Menjalankan Simulasi 10 Putaran LinUCB",
    )

    # -------------------------------------------------------------------------
    # 2. Eksperimen 2: Discounted LinUCB (Non-Stasioner, gamma = 0.85)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("🟢 [FASE 2] PENYIAPAN & EKSEKUSI EKSPERIMEN 2: DISCOUNTED LINUCB (D-LINUCB)")
    print("=" * 88)
    run_cmd(f'bash "{SETUP_SCRIPT}" "{RPC_URL}"', "Deploy Kontrak Bersih untuk D-LinUCB")

    run_cmd(
        f'python3 "{SIM_SCRIPT}" '
        f'--algorithm d-linucb --gamma 0.85 --alpha 0.50 --rounds 10 --output-log "{DLINUCB_OUT}"',
        "Menjalankan Simulasi 10 Putaran Discounted LinUCB (gamma=0.85)",
    )

    # -------------------------------------------------------------------------
    # 3. Analisis Hasil Komparasi
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("📊 [FASE 3] RINGKASAN METRIK KOMPARATIF LINUCB VS D-LINUCB")
    print("=" * 88)

    linucb_data = json.loads(LINUCB_OUT.read_text())
    dlinucb_data = json.loads(DLINUCB_OUT.read_text())

    l_diag = linucb_data["diagnostics"]
    d_diag = dlinucb_data["diagnostics"]

    print(f"| Metrik Evaluasi                  | LinUCB (Stasioner)    | D-LinUCB (gamma = 0.85) |")
    print(f"| :------------------------------- | :-------------------- | :---------------------- |")
    print(f"| Total Putaran Interaksi (T)      | {l_diag['total_rounds']:<21} | {d_diag['total_rounds']:<23} |")
    print(f"| Akumulasi Imbalan (Cumulative)   | {l_diag['cumulative_reward']:<+21.4f} | {d_diag['cumulative_reward']:<+23.4f} |")
    print(f"| Rata-rata Imbalan / Tx           | {l_diag['mean_reward']:<+21.4f} | {d_diag['mean_reward']:<+23.4f} |")
    print(f"| Total Kerentanan Teridentifikasi | {len(linucb_data['discovered_vulnerabilities']):<21} | {len(dlinucb_data['discovered_vulnerabilities']):<23} |")
    print(f"| Adaptasi Pasca-Drain (Round 9-10)| Evaluated in Report   | Evaluated in Report     |")
    print("=" * 88)


if __name__ == "__main__":
    main()

