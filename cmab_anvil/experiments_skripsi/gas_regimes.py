"""
Modul Pembangkitan Dinamika Harga Gas EVM (Kondisi 2 Eksperimen Skripsi).

Menyediakan dua rezim dinamika harga gas di jaringan simulator Anvil:
  - Rezim 1: Jaringan Stabil (Low Volatility, 15–25 Gwei)
  - Rezim 2: Jaringan Fluktuatif & Kongesti Mendadak (High Volatility & Gas Spikes, 20–200 Gwei)
Menggunakan Common Random Numbers (CRN) untuk menjamin replikasi adil antar kebijakan.
"""

from typing import List, Dict, Any, Optional
import numpy as np


def generate_gas_series(
    regime: str,
    rounds: int,
    seed: int = 42,
) -> List[float]:
    """
    Membangkitkan deret harga gas (dalam Gwei) untuk putaran transaksi t = 1, ..., T.

    Parameters
    ----------
    regime : str
        Pilihan rezim:
          - 'stable' (Rezim 1: Low Volatility, 15 - 25 Gwei)
          - 'volatile_spikes' (Rezim 2: High Volatility & Gas Spikes, 20 - 200 Gwei)
          - 'static' (Baseline kontrol: 1.0 Gwei)
    rounds : int
        Jumlah putaran horizon T.
    seed : int
        Seed acak deterministik (CRN).

    Returns
    -------
    List[float]
        Deret harga base fee per gas dalam satuan Gwei (dibulatkan 2 desimal).
    """
    if regime == "static":
        return [1.0] * rounds

    rng = np.random.default_rng(seed)
    t_arr = np.arange(1, rounds + 1, dtype=np.float64)

    if regime in ("stable", "low_volatility", "normal"):
        # Rezim 1: Jaringan Stabil (Tabel 3.1 Skripsi)
        # Log-normal EIP-1559 (mu = ln(15), sigma = 0.35), clipped [8.0, 32.0] Gwei
        samples = rng.lognormal(mean=np.log(15.0), sigma=0.35, size=rounds)
        clipped = np.clip(np.round(samples, 2), 8.0, 32.0)
        return [float(x) for x in clipped]

    elif regime in ("volatile_spikes", "high_volatility", "worst_case"):
        # Rezim 2: Fluktuatif & Lonjakan (Tabel 3.1 Skripsi)
        # Basis Uniform [60, 140] Gwei + peluang 25% lonjakan Pareto (alpha = 1.8 * 80)
        base = rng.uniform(60.0, 140.0, size=rounds)
        has_spike = rng.uniform(0.0, 1.0, size=rounds) < 0.25
        spikes = has_spike * (rng.pareto(a=1.8, size=rounds) * 80.0)
        clipped = np.clip(np.round(base + spikes, 2), 60.0, 500.0)
        return [float(x) for x in clipped]

    else:
        raise ValueError(
            f"Rezim gas tidak dikenal: '{regime}'. Pilih 'stable', 'volatile_spikes', atau 'static'."
        )


def get_gas_regime_metadata(regime: str) -> Dict[str, Any]:
    """Mengembalikan metadata deskriptif untuk pelaporan tabel skripsi."""
    if regime in ("stable", "low_volatility"):
        return {
            "regime_id": "REGIME_1",
            "name": "Jaringan Stabil (Low Volatility)",
            "range_gwei": "15 - 25 Gwei",
            "mean_gwei": "~20.0 Gwei",
            "characteristic": "Volatilitas rendah, variasi sempit sinusoidal + Gaussian noise",
        }
    elif regime in ("volatile_spikes", "high_volatility"):
        return {
            "regime_id": "REGIME_2",
            "name": "Jaringan Fluktuatif & Kongesti Mendadak (High Volatility & Gas Spikes)",
            "range_gwei": "20 - 200 Gwei",
            "mean_gwei": "~65.0 Gwei (Peak up to 200 Gwei)",
            "characteristic": "Gelombang burst + 25% lonjakan ekstrem mendadak (simulasi gas war DeFi)",
        }
    else:
        return {
            "regime_id": "REGIME_STATIC",
            "name": "Static Control Baseline",
            "range_gwei": "1.0 Gwei",
            "mean_gwei": "1.0 Gwei",
            "characteristic": "Konstan tanpa fluktuasi",
        }

