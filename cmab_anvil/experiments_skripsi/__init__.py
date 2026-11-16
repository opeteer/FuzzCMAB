"""
Eksperimen Skripsi: Evaluasi Ketahanan dan Efisiensi D-LinCBwK pada Smart Contract EVM.

Modul ini memfasilitasi pengujian faktorial melintasi 3 kondisi:
  1. Kompleksitas Kedalaman State Kontrak (Shallow, Medium, Deep)
  2. Dinamika Harga Gas Anvil (Rezim 1 Stabil vs Rezim 2 Fluktuatif & Spikes)
  3. Variasi Anggaran Modal Gas B (Tight, Moderate, Loose)
Menyandingkan 5 kebijakan:
  - UniformRandom (Forge Fuzz Baseline)
  - LinUCB (Stationary Unconstrained)
  - D-LinUCB (Discounted Unconstrained)
  - LinCBwK (Stationary Knapsack)
  - D-LinCBwK (Discounted Knapsack - Proposed)
"""

