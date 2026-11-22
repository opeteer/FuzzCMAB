#!/usr/bin/env python3
"""
Universal Simulation Runner: Unprivileged Contextual Multi-Armed Bandit
(LinUCB & Discounted LinUCB) running `Agen_Vault_Custody` on Anvil.
"""

import argparse
import json
from pathlib import Path
import sys

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

from cmab_anvil.Agen_Vault_Custody import (
    VaultCustodyEnv,
    VaultCustodyCMABAgent,
    CONTEXT_DIM,
    FEATURE_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Unprivileged CMAB Simulation (LinUCB / Discounted LinUCB) on Anvil"
    )
    parser.add_argument(
        "--rpc-url",
        type=str,
        default="http://127.0.0.1:8545",
        help="Anvil JSON-RPC endpoint URL",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["uniform", "linucb", "d-linucb", "lincbwk", "d-lincbwk", "d-linucbwk"],
        default="d-lincbwk",
        help="CMAB algorithm strategy: 'uniform', 'linucb', 'd-linucb', 'lincbwk', or 'd-lincbwk' ('d-linucbwk')",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=25,
        help="Number of CMAB interaction rounds T",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=1.50,
        help="Initial gas capital budget B in ETH",
    )
    parser.add_argument(
        "--gas-regime",
        type=str,
        choices=["static", "stable", "volatile_spikes"],
        default="stable",
        help="Gas price fluctuation regime: 'static', 'stable', or 'volatile_spikes'",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.82,
        help="Discount factor gamma in (0, 1] for Discounted policies",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Exploration confidence multiplier alpha",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=1.0,
        help="Ridge regularization parameter lambda",
    )
    parser.add_argument(
        "--theoretical-beta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Russac et al. (2019) analytical confidence radius beta_t in D-LinUCB",
    )
    parser.add_argument(
        "--decay-all-arms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply synchronous exponential decay to all arms each round",
    )
    parser.add_argument(
        "--peer-traffic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simulate stochastic external party transactions (peer deposits/withdrawals/transfers)",
    )
    parser.add_argument(
        "--output-log",
        type=str,
        default=str(RESULTS_DIR / "cmab_simulation_results.json"),
        help="Path to save simulation telemetry JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    algo_map = {
        "uniform": "UniformRandom (Baseline)",
        "linucb": "Disjoint LinUCB",
        "d-linucb": "Discounted LinUCB (D-LinUCB)",
        "lincbwk": "Stationary LinCBwK",
        "d-lincbwk": "Discounted LinCBwK (D-LinUCBwK)",
        "d-linucbwk": "Discounted LinCBwK (D-LinUCBwK)",
    }
    algo_label = algo_map.get(args.algorithm, args.algorithm)

    print("=" * 96)
    print(f"🧠 SIMULASI CMAB ({algo_label}) - AGEN_VAULT_CUSTODY PADA ANVIL")
    print("=" * 96)

    env = VaultCustodyEnv(
        rpc_url=args.rpc_url,
        agent_ref_budget_eth=args.budget,
        gas_regime=args.gas_regime,
        horizon_t=args.rounds,
        enable_peer_traffic=args.peer_traffic,
        seed=42,
    )

    # 1. Verify Strict Zero-Privilege Standing via Universal RPC Client
    standing = env.verify_unprivileged_standing()
    print("[*] Verifikasi Kedudukan Tanpa Hak Istimewa (Zero-Privilege Check):")
    print(f"    - Spesifikasi Agen   : {standing['agent_name']}")
    print(f"    - Alamat EOA Agen    : {standing['agent_address']} (Pengguna Biasa)")
    print(f"    - Owner Kontrak Vault: {standing['vault_owner']} (Bukan Agen)")
    print(f"    - Alamat Vault       : {standing['vault_address']}")
    print(f"    - Alamat MockToken   : {standing['token_address']}")
    print(f"    - Status Kedudukan   : {standing['standing']}")
    print(f"    - Rezim Harga Gas    : {args.gas_regime} | Anggaran Modal B = {args.budget:.2f} ETH")
    print(f"    - Transaksi Eksternal: {'Aktif (CRN Peer Traffic)' if args.peer_traffic else 'Nonaktif'}")

    # 2. Instantiate Agen_Vault_Custody with selected algorithm
    agent = VaultCustodyCMABAgent(
        env=env,
        algorithm=args.algorithm,
        alpha=args.alpha,
        l2_lambda=args.l2_lambda,
        gamma=args.gamma,
        initial_budget=args.budget,
        total_horizon=args.rounds,
        use_theoretical_beta=args.theoretical_beta,
        decay_all_arms=args.decay_all_arms,
        seed=42,
    )

    print(f"\n[*] Konfigurasi Model CMAB & Strategi {agent.policy.name}:")
    print(
        f"    - Dimensi Konteks (d): {CONTEXT_DIM} fitur on-chain publik "
        f"({', '.join(FEATURE_NAMES[:4])}, ...)"
    )
    print(f"    - Jumlah Lengan (K)  : {env.n_arms} strategi transaksi EOA")
    print(
        f"    - Hyperparameter     : gamma = {args.gamma}, alpha = {args.alpha}, "
        f"lambda = {args.l2_lambda}, budget B = {args.budget:.2f} ETH, horizon T = {args.rounds}"
    )
    print("-" * 96)

    history = []

    for t in range(1, args.rounds + 1):
        if env.remaining_budget_eth <= env.min_capital_threshold_eth:
            print(f"[!] Anggaran modal gas habis pada putaran ke-{t} (B = {env.remaining_budget_eth:.4f} ETH).")
            break

        selection, transition = agent.act_and_learn()
        reward = transition.reward_breakdown.total_reward
        status_tag = "SUCCESS" if transition.tx_succeeded else "REVERT "

        print(
            f"[Round {t:02d}/{args.rounds:02d}] "
            f"Gas={transition.gas_price_gwei:5.1f}Gwei | Ext={transition.external_event:<19} | "
            f"Arm #{selection.arm_id} ({transition.arm_spec.name:<27}) | "
            f"Tx={status_tag} | Rew={reward:.2f} ({transition.reward_breakdown.event_partition}) | "
            f"Cost={transition.tx_cost_eth:.4f} | B_t={transition.remaining_budget_eth:.4f} | "
            f"VaultETH={transition.next_snapshot.vault_eth_balance:.2f}"
        )
        for anomaly in transition.reward_breakdown.detected_anomalies:
            print(f"    [!] TEMUAN ANOMALI / KERENTANAN: {anomaly}")

        history.append(
            {
                "round": t,
                "gas_price_gwei": transition.gas_price_gwei,
                "external_event": transition.external_event,
                "arm_id": selection.arm_id,
                "arm_name": transition.arm_spec.name,
                "ucb_score": selection.ucb_score,
                "expected_reward": selection.expected_reward,
                "uncertainty_bonus": selection.uncertainty_bonus,
                "tx_succeeded": transition.tx_succeeded,
                "reward": reward,
                "cost_eth": transition.tx_cost_eth,
                "remaining_budget_eth": transition.remaining_budget_eth,
                "reward_breakdown": transition.reward_breakdown.to_dict(),
                "context": transition.context_vector.tolist(),
                "vault_eth_after": transition.next_snapshot.vault_eth_balance,
                "agent_eoa_eth_after": transition.next_snapshot.agent_eoa_eth_balance,
                "agent_vault_stake_after": transition.next_snapshot.agent_vault_balance,
                "vault_solvent_after": transition.next_snapshot.vault_is_solvent,
            }
        )

    diag = agent.policy.get_diagnostics()
    print("-" * 96)
    print(f"📈 RINGKASAN EVALUASI AGEN_VAULT_CUSTODY ({agent.policy.name}):")
    print(f"    - Total Putaran Bertahan  : {diag['total_rounds']}/{args.rounds}")
    print(f"    - Akumulasi Reward        : {diag['cumulative_reward']:+.4f}")
    print(f"    - Rata-rata Reward / Tx   : {diag['mean_reward']:+.4f}")
    print(f"    - Sisa Anggaran Gas (B_T) : {env.remaining_budget_eth:.4f} ETH")
    print(f"    - Total Anomali Terdeteksi: {len(agent.discovered_vulnerabilities)}")
    print("    - Distribusi Pemilihan Lengan & Bobot Parameter:")
    effective_pulls = diag.get("effective_pulls")
    theta_norms = diag.get("theta_l2_norms", [0.0] * env.n_arms)
    for arm in env.arms:
        pulls = diag["arm_counts"][arm.arm_id]
        norm_theta = theta_norms[arm.arm_id] if arm.arm_id < len(theta_norms) else 0.0
        if effective_pulls is not None:
            ess = effective_pulls[arm.arm_id]
            print(
                f"        [{arm.arm_id}] {arm.name:<28}: {pulls:3d}x dipilih "
                f"(ESS={ess:.3f}) | ||theta_{arm.arm_id}||_2 = {norm_theta:.4f}"
            )
        else:
            print(
                f"        [{arm.arm_id}] {arm.name:<28}: {pulls:3d}x dipilih | "
                f"||theta_{arm.arm_id}||_2 = {norm_theta:.4f}"
            )
    print("=" * 96)

    out_path = Path(args.output_log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "standing": standing,
                "diagnostics": diag,
                "discovered_vulnerabilities": agent.discovered_vulnerabilities,
                "feature_names": FEATURE_NAMES,
                "history": history,
            },
            indent=2,
        )
    )
    if hasattr(agent.policy, "save_checkpoint"):
        ckpt_name = f"{args.algorithm.replace('-', '_')}_checkpoint.json"
        agent.policy.save_checkpoint(out_path.parent / ckpt_name)


if __name__ == "__main__":
    main()
