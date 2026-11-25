#!/usr/bin/env python3
"""
Automated 5-Policy Comparative Evaluation Runner for Agen_Vault_Custody on Anvil.

Executes all 5 benchmark policies under identical Common Random Numbers (CRN)
starting conditions, gas price fluctuation regime, and initial gas capital budget B:
  1. UniformRandom (Forge Fuzz Baseline)
  2. LinUCB (Stationary Unconstrained: gamma = 1.0, lambda_t = 0)
  3. D-LinUCB (Discounted Unconstrained: gamma = 0.82, lambda_t = 0)
  4. LinCBwK (Stationary Knapsack: gamma = 1.0, dynamic lambda_t)
  5. D-LinCBwK (Discounted Knapsack - Proposed: gamma = 0.82, dynamic lambda_t)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
SIM_SCRIPT = CMAB_ROOT / "tests" / "run_cmab_simulation.py"

POLICIES: List[Tuple[str, str]] = [
    ("uniform", "UniformRandom"),
    ("linucb", "LinUCB (gamma=1.0)"),
    ("d-linucb", "D-LinUCB (gamma=0.82)"),
    ("lincbwk", "LinCBwK (gamma=1.0)"),
    ("d-lincbwk", "D-LinCBwK (Proposed)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5-Policy Comparative Experiment for Agen_Vault_Custody on Anvil"
    )
    parser.add_argument("--rpc-url", type=str, default="http://127.0.0.1:8545")
    parser.add_argument("--rounds", type=int, default=15, help="Horizon T rounds per policy")
    parser.add_argument("--budget", type=float, default=1.50, help="Initial gas budget B (ETH)")
    parser.add_argument(
        "--gas-regime",
        type=str,
        choices=["static", "stable", "volatile_spikes"],
        default="volatile_spikes",
    )
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--gamma", type=float, default=0.82)
    return parser.parse_args()


def run_cmd(cmd: str, desc: str) -> None:
    print(f"\n[*] {desc}...")
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(f"[!] Error executing: {cmd}\nOutput:\n{p.stdout}")
        sys.exit(1)
    print(p.stdout)


def main() -> None:
    args = parse_args()
    print("=" * 96)
    print("🔬 EKSPERIMEN KOMPARATIF 5 KEBIJAKAN CMAB PADA ANVIL (AGEN_VAULT_CUSTODY)")
    print(
        f"   Rezim Gas: {args.gas_regime} | Anggaran Modal B = {args.budget:.2f} ETH | "
        f"Horizon T = {args.rounds} | Alpha = {args.alpha} | Gamma = {args.gamma}"
    )
    print("=" * 96)

    summary_rows: List[Dict[str, object]] = []

    for algo_flag, algo_label in POLICIES:
        out_file = RESULTS_DIR / f"{algo_flag.replace('-', '_')}_simulation_results.json"
        run_cmd(f'bash "{SETUP_SCRIPT}" "{args.rpc_url}"', f"Deploy Kontrak Bersih untuk {algo_label}")
        run_cmd(
            f'python3 "{SIM_SCRIPT}" '
            f'--algorithm {algo_flag} --alpha {args.alpha} --gamma {args.gamma} '
            f'--rounds {args.rounds} --budget {args.budget} --gas-regime {args.gas_regime} '
            f'--output-log "{out_file}"',
            f"Menjalankan Simulasi {algo_label}",
        )
        data = json.loads(out_file.read_text())
        diag = data["diagnostics"]
        hist = data.get("history", [])
        rem_b = hist[-1]["remaining_budget_eth"] if hist else args.budget
        burned = max(1e-6, args.budget - rem_b)
        e4_hits = sum(
            1 for h in hist if h.get("reward_breakdown", {}).get("event_partition") == "E4"
        )
        summary_rows.append(
            {
                "label": algo_label,
                "rounds": diag["total_rounds"],
                "cum_reward": diag["cumulative_reward"],
                "mean_reward": diag["mean_reward"],
                "e4_hits": e4_hits,
                "anomalies": len(data.get("discovered_vulnerabilities", [])),
                "eff_gas": diag["cumulative_reward"] / burned,
                "rem_budget": rem_b,
            }
        )

    print("\n" + "=" * 96)
    print("📊 RINGKASAN METRIK KOMPARATIF 5 KEBIJAKAN (AGEN_VAULT_CUSTODY)")
    print("=" * 96)
    print(
        f"| {'Kebijakan':<22} | {'Putaran':<8} | {'Total Rew':<10} | {'Mean Rew':<9} | "
        f"{'Vuln (E4)':<9} | {'Eff_gas':<9} | {'Sisa B (ETH)':<12} |"
    )
    print(f"| :{'-'*21} | :{'-'*7} | :{'-'*9} | :{'-'*8} | :{'-'*8} | :{'-'*8} | :{'-'*11} |")
    for row in summary_rows:
        print(
            f"| {str(row['label']):<22} | "
            f"{int(row['rounds']):2d}/{args.rounds:<5d} | "
            f"{float(row['cum_reward']):+10.4f} | "
            f"{float(row['mean_reward']):+9.4f} | "
            f"{int(row['e4_hits']):9d} | "
            f"{float(row['eff_gas']):9.2f} | "
            f"{float(row['rem_budget']):12.4f} |"
        )
    print("=" * 96)


if __name__ == "__main__":
    main()

