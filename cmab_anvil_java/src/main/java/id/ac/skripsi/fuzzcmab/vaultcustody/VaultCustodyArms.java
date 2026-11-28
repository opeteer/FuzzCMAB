package id.ac.skripsi.fuzzcmab.vaultcustody;

import id.ac.skripsi.fuzzcmab.common.Arm;

import java.util.List;

/**
 * Katalog Ruang Aksi / Lengan (A_Vault = {0, ..., 7}) untuk Agen_Vault_Custody.
 *
 * Setiap objek Arm adalah transaksi EOA agen pengguna biasa yang menargetkan
 * VulnerableVault atau MockToken sesuai distribusi fungsi dataset SC-Bench Vault_and_Liquidity.
 */
public final class VaultCustodyArms {

    private VaultCustodyArms() {
    }

    public static List<Arm> buildVaultCustodyArmCatalog() {
        return List.of(
                new Arm(
                        0,
                        "vault_deposit_small",
                        "VulnerableVault",
                        "deposit()",
                        0.25,
                        "Deposit 0.25 ETH into VulnerableVault to establish user stake (SC-Bench deposit standard)."
                ),
                new Arm(
                        1,
                        "vault_deposit_medium",
                        "VulnerableVault",
                        "deposit()",
                        1.00,
                        "Deposit 1.00 ETH into VulnerableVault to increase stake & vault liquidity (SC-Bench deposit standard)."
                ),
                new Arm(
                        2,
                        "vault_withdraw",
                        "VulnerableVault",
                        "withdraw()",
                        0.0,
                        "Call withdraw() to reclaim agent's recorded balance from Vault (SC-Bench withdraw standard)."
                ),
                new Arm(
                        3,
                        "vault_fallback_transfer",
                        "VulnerableVault",
                        "",
                        0.10,
                        "Send 0.10 ETH directly via receive() fallback to test accounting parity."
                ),
                new Arm(
                        4,
                        "vault_deposit_zero_probe",
                        "VulnerableVault",
                        "deposit()",
                        0.0,
                        "Boundary probe: call deposit() with 0 ETH (tests require(msg.value > 0))."
                ),
                new Arm(
                        5,
                        "vault_emergency_withdraw_probe",
                        "VulnerableVault",
                        "emergencyWithdraw()",
                        0.0,
                        "Probe conditional access control flaw in emergencyWithdraw() (SC-Bench emergencyWithdraw standard)."
                ),
                new Arm(
                        6,
                        "token_mint_user",
                        "MockToken",
                        "mint(address,uint256)",
                        0.0,
                        "Call public mint(agent, 250 MCK) on MockToken contract (SC-Bench mint standard)."
                ),
                new Arm(
                        7,
                        "token_transfer_vault",
                        "MockToken",
                        "transfer(address,uint256)",
                        0.0,
                        "Transfer 50 MCK tokens to VulnerableVault address (SC-Bench transfer standard)."
                )
        );
    }
}
