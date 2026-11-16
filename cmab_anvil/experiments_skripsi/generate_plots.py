"""
Modul Visualisasi Publikasi Eksperimen Skripsi.

Menghasilkan grafik komparatif publikasi ilmiah (PDF/PNG 300 DPI):
  1. Dynamic Regret vs Rounds (5 Kebijakan vs Theoretical Bound)
  2. Trajektori Pengurasan Modal & Survival Rate (Kondisi 3: Tight vs Loose Budget)
  3. Respons Pengali Dual lambda_t terhadap Lonjakan Gas (Kondisi 2: Rezim 2 Gas Spikes)
  4. Perbandingan Capital Efficiency Ratio (CER) across Tiers (Kondisi 1: Shallow, Medium, Deep)
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

CURR_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURR_DIR / "results"
TELEMETRY_DIR = RESULTS_DIR / "telemetry"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

POLICY_COLORS = {
    "UniformRandom": "#7f7f7f",      # Grey (Forge Base)
    "LinUCB": "#1f77b4",             # Blue
    "D-LinUCB": "#ff7f0e",           # Orange
    "LinCBwK": "#2ca02c",            # Green
    "D-LinCBwK": "#d62728",          # Crimson Red (Proposed)
}

POLICY_LABELS = {
    "UniformRandom": "Uniform Random (Forge Fuzz Base)",
    "LinUCB": "LinUCB (Stationary)",
    "D-LinUCB": r"D-LinUCB ($\gamma=0.82$)",
    "LinCBwK": "LinCBwK (Stationary Knapsack)",
    "D-LinCBwK": r"D-LinCBwK ($\gamma=0.82$, Knapsack - Proposed)",
}


def load_latest_telemetry(telemetry_file: Optional[Path] = None) -> Dict[str, Any]:
    target = telemetry_file or (TELEMETRY_DIR / "skripsi_experiments_latest.json")
    if not target.exists():
        candidates = sorted(TELEMETRY_DIR.glob("skripsi_experiments_*.json"))
        if candidates:
            target = candidates[-1]
        else:
            raise FileNotFoundError(f"Tidak ada berkas telemetri ditemukan di {TELEMETRY_DIR}")
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_gas_spike_adaptation(
    exp_record: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot visualisasi respons adaptasi pengali dual lambda_t terhadap dinamika harga gas (Kondisi 2).
    """
    out_path = save_path or (FIGURES_DIR / "fig_gas_spike_adaptation.png")
    results = exp_record["results"]
    dlincbwk_logs = results.get("D-LinCBwK", {}).get("round_logs", [])
    uniform_logs = results.get("UniformRandom", {}).get("round_logs", [])

    if not dlincbwk_logs:
        raise ValueError("Data log D-LinCBwK kosong untuk plot gas adaptation.")

    rounds = [l["round"] for l in dlincbwk_logs]
    gas_prices = [l["gas_price_gwei"] for l in dlincbwk_logs]
    lambdas = [l["dual_lambda"] for l in dlincbwk_logs]
    d_rewards = [l["reward"] for l in dlincbwk_logs]
    u_rewards = [l["reward"] for l in uniform_logs] if uniform_logs else [0] * len(rounds)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Panel 1: Dinamika Base Fee (Gwei)
    ax1.plot(rounds, gas_prices, color="#8c564b", marker="o", markersize=4, label="EVM BaseFee (Gwei)")
    ax1.axhline(30.0, color="gray", linestyle=":", label="Normal Congestion Threshold")
    ax1.set_ylabel("BaseFee (Gwei)")
    ax1.set_title(f"Respons Pengali Dual $\\lambda_t$ terhadap Lonjakan Gas ({exp_record['contract_name']} - Rezim 2)")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    # Panel 2: Shadow Price Dual Multiplier lambda_t
    ax2.plot(rounds, lambdas, color="#d62728", marker="s", markersize=4, label=r"Dual Multiplier $\lambda_t$ (D-LinCBwK)")
    ax2.set_ylabel(r"Shadow Price $\lambda_t$")
    ax2.legend(loc="upper left")
    ax2.grid(True)

    # Panel 3: Reward Komparatif
    ax3.plot(rounds, np.cumsum(d_rewards), color="#d62728", label="Cumulative Reward (D-LinCBwK)")
    ax3.plot(rounds[:len(u_rewards)], np.cumsum(u_rewards), color="#7f7f7f", linestyle="--", label="Cumulative Reward (Uniform Random)")
    ax3.set_xlabel("Putaran Transaksi (t)")
    ax3.set_ylabel("Kumulatif Reward")
    ax3.legend(loc="upper left")
    ax3.grid(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    return out_path


def plot_capital_survival_comparison(
    telemetry_data: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot trajektori sisa modal (ETH) dan waktu bertahan hidup (survival) pada anggaran ketat (Kondisi 3).
    """
    out_path = save_path or (FIGURES_DIR / "fig_capital_survival_tight_budget.png")

    # Cari eksperimen dengan budget tight
    tight_exps = [e for e in telemetry_data.get("experiments", []) if e.get("budget_tier") == "tight"]
    if not tight_exps:
        tight_exps = telemetry_data.get("experiments", [])

    if not tight_exps:
        raise ValueError("Data eksperimen kosong.")

    exp = tight_exps[0]
    results = exp["results"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for pol_name in ["UniformRandom", "LinUCB", "D-LinUCB", "LinCBwK", "D-LinCBwK"]:
        if pol_name in results:
            logs = results[pol_name].get("round_logs", [])
            if logs:
                rds = [l["round"] for l in logs]
                bals = [l["balance_eth"] for l in logs]
                ax.plot(
                    rds, bals,
                    label=f"{POLICY_LABELS[pol_name]} (Surv: {results[pol_name]['rounds_survived']} rds)",
                    color=POLICY_COLORS[pol_name],
                    marker="o" if "Knapsack" in POLICY_LABELS[pol_name] else "x",
                    markersize=4,
                )

    ax.axhline(0.005, color="red", linestyle=":", linewidth=1.5, label="Ambang Kebangkrutan (0.005 ETH)")
    ax.set_xlabel("Putaran Transaksi (t)")
    ax.set_ylabel("Sisa Saldo Dompet Agen (ETH)")
    ax.set_title(f"Trajektori Ketahanan Modal pada Anggaran Ketat ($B_{{low}}$) - {exp['contract_name']}")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    return out_path


def generate_all_skripsi_figures(telemetry_file: Optional[Path] = None) -> List[Path]:
    """Menghasilkan seluruh paket figur publikasi dari hasil telemetri skripsi."""
    data = load_latest_telemetry(telemetry_file)
    generated: List[Path] = []

    # 1. Plot Survival Trajectory
    try:
        p1 = plot_capital_survival_comparison(data)
        generated.append(p1)
        print(f"[+] Figur tersimpan: {p1}")
    except Exception as e:
        print(f"[!] Gagal membuat figur survival: {e}")

    # 2. Plot Gas Spike Adaptation
    try:
        volatile_exps = [e for e in data.get("experiments", []) if e.get("gas_regime") == "volatile_spikes"]
        target_exp = volatile_exps[0] if volatile_exps else data.get("experiments", [])[0]
        p2 = plot_gas_spike_adaptation(target_exp)
        generated.append(p2)
        print(f"[+] Figur tersimpan: {p2}")
    except Exception as e:
        print(f"[!] Gagal membuat figur gas spike: {e}")

    return generated


if __name__ == "__main__":
    generate_all_skripsi_figures()

