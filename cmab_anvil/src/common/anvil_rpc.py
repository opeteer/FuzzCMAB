"""
Universal Regular User JSON-RPC / Cast Client for the Anvil Simulator.

Enforces the Regular User Agent Model (Agen Pengguna Biasa) across any CMAB Agent:
  1. Uses an ordinary Externally Owned Account (EOA) identity (`AGENT_ADDR`).
  2. Cryptographically and on-chain verifies that `AGENT_ADDR` is NOT the contract owner.
  3. Prohibits calling any privileged state-mutating `anvil_*`, `evm_*`, or `hardhat_*` cheatcode methods.
  4. Implements a 4-Tiered ABI-Agnostic Context Resolution Engine:
       - Tier 0: Intrinsic ABI-free EVM queries (`eth_getBalance`, `eth_getCode`, `eth_blockNumber`, opcode traces).
       - Tier 1: Multi-standard ABI probing (ERC-4626, ERC-20, Vault, Staking).
       - Tier 2: Dynamic ABI Introspection (automatic type & lexical ranking from `solc` / Foundry ABI JSON).
       - Tier 3: Pure-Bytecode EVM Storage & Keccak256 Mapping Slot Inference (`keccak256(abi.encode(agent, p))`)
         for contracts with `private`/`internal` state variables and no public getters.
"""

import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.request


# =============================================================================
# Fast Pure-Python Ethereum Keccak-256 (0x01 padding) for Mapping Slot Inference
# =============================================================================
_KECCAK_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

_KECCAK_ROT = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]


def _rol64(val: int, shift: int) -> int:
    return ((val << shift) | (val >> (64 - shift))) & 0xFFFFFFFFFFFFFFFF


def keccak256_bytes(data: bytes) -> bytes:
    """Compute Ethereum Keccak-256 hash in pure Python with zero subprocess overhead."""
    rate = 136
    padded = bytearray(data)
    padded.append(0x01)
    while len(padded) % rate != 0:
        padded.append(0x00)
    padded[-1] |= 0x80
    state = [[0] * 5 for _ in range(5)]
    for offset in range(0, len(padded), rate):
        for i in range(rate // 8):
            x, y = i % 5, i // 5
            lane = int.from_bytes(padded[offset + 8 * i : offset + 8 * (i + 1)], "little")
            state[x][y] ^= lane
        for r in range(24):
            c = [state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4] for x in range(5)]
            d = [c[(x - 1) % 5] ^ _rol64(c[(x + 1) % 5], 1) for x in range(5)]
            for x in range(5):
                for y in range(5):
                    state[x][y] ^= d[x]
            b = [[0] * 5 for _ in range(5)]
            for x in range(5):
                for y in range(5):
                    b[y][(2 * x + 3 * y) % 5] = _rol64(state[x][y], _KECCAK_ROT[x][y])
            for x in range(5):
                for y in range(5):
                    state[x][y] = b[x][y] ^ ((~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y])
            state[0][0] ^= _KECCAK_RC[r]
    out = bytearray()
    for y in range(5):
        for x in range(5):
            out.extend(state[x][y].to_bytes(8, "little"))
            if len(out) >= 32:
                return bytes(out[:32])
    return bytes(out[:32])


def compute_solidity_mapping_slot(user_address: str, mapping_base_slot: int) -> str:
    """
    Compute the deterministic Solidity storage slot for `mapping(address => uint256)`
    at base slot `p`: `keccak256(abi.encode(address, uint256(p)))`.
    """
    clean_addr = user_address.lower().replace("0x", "").rjust(64, "0")
    clean_slot = hex(mapping_base_slot)[2:].rjust(64, "0")
    encoded = bytes.fromhex(clean_addr + clean_slot)
    return "0x" + keccak256_bytes(encoded).hex()


class RegularUserAnvilClient:
    """
    Universal RPC client for interacting with Anvil strictly as a regular user agent (agen pengguna biasa),
    equipped with 4-Tiered ABI Resolution (Multi-ABI Probing, Dynamic ABI Introspection,
    Keccak256 Mapping Slot Inference) and Hybrid EVM `SLOAD`/`SSTORE` trace analysis.
    """

    # Anvil Account #0: Deployer / Owner (Strictly distinct from the CMAB Agent)
    DEPLOYER_ADDR: str = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"

    # Anvil Account #1: Peer Normal User (for optional background traffic simulation)
    PEER_USER_ADDR: str = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
    PEER_USER_PK: str = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"

    # Anvil Account #2: Regular User CMAB Agent (Agen Pengguna Biasa)
    DEFAULT_AGENT_ADDR: str = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"
    DEFAULT_AGENT_PK: str = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"

    # State-mutating cheatcodes remain strictly forbidden
    FORBIDDEN_RPC_PREFIXES = ("anvil_", "evm_", "hardhat_")

    TX_HASH_REGEX = re.compile(r"0x[a-fA-F0-9]{64}")

    # Tier 1 Default Standard ABI Signatures
    STANDARD_GLOBAL_LIABILITY_SIGS: List[str] = [
        "totalDeposited()(uint256)",
        "totalAssets()(uint256)",
        "totalStaked()(uint256)",
        "totalSupply()(uint256)",
        "poolBalance()(uint256)",
        "totalBalance()(uint256)",
        "getTotalDeposits()(uint256)",
    ]

    STANDARD_USER_STAKE_SIGS: List[str] = [
        "balances(address)(uint256)",
        "balanceOf(address)(uint256)",
        "stakes(address)(uint256)",
        "deposits(address)(uint256)",
        "shares(address)(uint256)",
        "userInfo(address)(uint256)",
        "getStake(address)(uint256)",
    ]

    def __init__(
        self,
        rpc_url: str = "http://127.0.0.1:8545",
        agent_address: Optional[str] = None,
        agent_private_key: Optional[str] = None,
        cast_bin: Optional[Path] = None,
        num_tracked_slots: int = 8,
    ):
        self.rpc_url = rpc_url
        self.agent_address = agent_address or self.DEFAULT_AGENT_ADDR
        self.agent_private_key = agent_private_key or self.DEFAULT_AGENT_PK
        self.cast_bin = (
            cast_bin
            if cast_bin is not None
            else Path.home() / ".foundry" / "bin" / "cast"
        )
        self.num_tracked_slots = num_tracked_slots

        # Tracks cumulative SLOAD slots and storage slot history per target contract
        self._observed_sload_slots: Dict[str, Set[str]] = {}
        self._last_storage_snapshot: Dict[str, Dict[str, str]] = {}

        # Tier 2: Dynamically discovered ABI getter signatures per contract address
        self._dynamic_global_getters: Dict[str, List[str]] = {}
        self._dynamic_user_getters: Dict[str, List[str]] = {}

        # Tier 3: Cached working mapping base slot p for `mapping(address => uint256)`
        self._inferred_mapping_slot: Dict[str, int] = {}

    def _json_rpc(self, method: str, params: List[Any]) -> Optional[Any]:
        """Direct lightweight JSON-RPC call for read-only queries and trace inspection."""
        for forbidden in self.FORBIDDEN_RPC_PREFIXES:
            if method.startswith(forbidden):
                raise PermissionError(
                    f"Regular User Standing Violation: Forbidden state-mutating cheatcode '{method}' blocked!"
                )
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        ).encode("utf-8")
        req = urllib.request.Request(
            self.rpc_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "error" in data:
                    return None
                return data.get("result")
        except Exception:
            return None

    def _run_cast(self, args: List[str]) -> Tuple[str, str, int]:
        for token in args:
            for forbidden in self.FORBIDDEN_RPC_PREFIXES:
                if token.startswith(forbidden):
                    raise PermissionError(
                        f"Regular User Standing Violation: Forbidden RPC cheatcode '{token}' blocked!"
                    )

        cmd = [str(self.cast_bin)] + args + ["--rpc-url", self.rpc_url]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15.0,
            )
            return proc.stdout.strip(), proc.stderr.strip(), proc.returncode
        except Exception as exc:
            return "", str(exc), 1

    @staticmethod
    def parse_uint256_ether(raw_output: str) -> float:
        """Convert cast uint256 wei output string into float ETH/Token units (1e18)."""
        if not raw_output:
            return 0.0
        first_token = raw_output.split()[0].strip()
        try:
            return float(int(first_token, 0)) / 1e18
        except ValueError:
            return 0.0

    # -------------------------------------------------------------------------
    # Tier 2: Dynamic ABI Introspection & Semantic Getter Discovery
    # -------------------------------------------------------------------------
    def register_contract_abi(
        self, contract_address: str, abi: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Inspect a contract's JSON ABI (from `solc` or Foundry artifact) and automatically
        discover & rank candidate getters for:
          1. Global Vault Liability / Pool TVL (`() -> uint256`)
          2. Agent Stake / User Balance (`(address) -> uint256` or tuple starting with `uint256`)
        """
        addr_key = contract_address.lower()
        global_candidates: List[Tuple[int, str]] = []
        user_candidates: List[Tuple[int, str]] = []

        global_pos_kw = (
            "totaldeposited", "totalassets", "totalstaked", "totalsupply",
            "poolbalance", "totalbalance", "tvl", "reserve", "liquidity",
            "staked", "deposited", "total", "pool", "balance", "supply", "funds",
        )
        global_neg_kw = (
            "decimal", "version", "chainid", "timestamp", "block", "owner",
            "fee", "rate", "index", "nonce", "count", "length", "time",
            "period", "duration", "lock", "min", "max", "cap", "ratio", "reward",
        )

        user_pos_kw = (
            "balances", "balanceof", "stakes", "stakeof", "deposits",
            "depositof", "shares", "shareof", "userinfo", "getstake",
            "userbalance", "locked", "claimable", "amount", "user", "account",
        )
        user_neg_kw = (
            "nonce", "whitelist", "blacklist", "allowed", "is", "has", "index",
        )

        for item in abi:
            if not isinstance(item, dict) or item.get("type") != "function":
                continue
            state_mut = item.get("stateMutability", "")
            is_const = item.get("constant", False)
            if state_mut not in ("view", "pure") and not is_const:
                continue

            fname = item.get("name", "")
            if not fname:
                continue
            fname_low = fname.lower()
            inputs = item.get("inputs", [])
            outputs = item.get("outputs", [])
            if not outputs:
                continue

            first_out_type = outputs[0].get("type", "")
            if not first_out_type.startswith("uint"):
                continue

            # Case 1: 0-argument getter `func()(uint256)`
            if len(inputs) == 0:
                if any(neg in fname_low for neg in global_neg_kw):
                    continue
                score = 0
                for idx, kw in enumerate(global_pos_kw):
                    if kw in fname_low:
                        score += (len(global_pos_kw) - idx) * 10
                if score > 0:
                    sig = f"{fname}()({first_out_type})"
                    global_candidates.append((score, sig))

            # Case 2: 1-argument address getter `func(address)(uint256)`
            elif len(inputs) == 1 and inputs[0].get("type") == "address":
                if any(neg in fname_low for neg in user_neg_kw):
                    continue
                score = 1  # Any `(address) -> uint*` is a valid structural candidate
                for idx, kw in enumerate(user_pos_kw):
                    if kw in fname_low:
                        score += (len(user_pos_kw) - idx) * 10
                sig = f"{fname}(address)({first_out_type})"
                user_candidates.append((score, sig))

        global_candidates.sort(key=lambda x: x[0], reverse=True)
        user_candidates.sort(key=lambda x: x[0], reverse=True)

        disc_global = [sig for _, sig in global_candidates]
        disc_user = [sig for _, sig in user_candidates]

        self._dynamic_global_getters[addr_key] = disc_global
        self._dynamic_user_getters[addr_key] = disc_user
        return {"global_getters": disc_global, "user_getters": disc_user}

    # -------------------------------------------------------------------------
    # Tier 3: Pure-Bytecode EVM Storage & Keccak256 Mapping Slot Inference
    # -------------------------------------------------------------------------
    def infer_global_liability_from_storage(
        self, contract_address: str, vault_eth_hint: float = 0.0
    ) -> float:
        """
        Tier 3 Fallback: When no public ABI getter exists for total deposits/assets,
        scan base storage slots `0x0`..`0x7` via `eth_getStorageAt` for 18-decimal wei amounts.
        """
        candidates: List[float] = []
        for slot_idx in range(self.num_tracked_slots):
            raw_hex = self.get_storage_at(contract_address, slot_idx)
            try:
                raw_int = int(raw_hex, 16)
                # Filter for plausible wei values [1e14 (0.0001 ETH), 1e26 (100M ETH)]
                # Note: 20-byte addresses are ~1.46e48 or when right-aligned in 32 bytes have non-wei entropy
                if 10**14 <= raw_int <= 10**26:
                    eth_val = float(raw_int) / 1e18
                    candidates.append(eth_val)
            except ValueError:
                continue

        if not candidates:
            return 0.0
        if vault_eth_hint > 0.0:
            # Pick the storage slot value closest in magnitude to the vault's ETH balance
            candidates.sort(key=lambda v: abs(v - vault_eth_hint))
        return candidates[0]

    def infer_mapping_balance_from_storage(
        self, contract_address: str, user_address: Optional[str] = None
    ) -> float:
        """
        Tier 3 Fallback: When a contract stores user balances in a `private` or non-standard
        `mapping(address => uint256)` without a known ABI getter, probe deterministic Solidity
        mapping slots `keccak256(abi.encode(user_address, p))` for p in 0..num_tracked_slots-1.
        """
        u_addr = user_address or self.agent_address
        addr_key = contract_address.lower()

        # Check cached mapping slot first if previously discovered
        if addr_key in self._inferred_mapping_slot:
            p_cached = self._inferred_mapping_slot[addr_key]
            slot_hash = compute_solidity_mapping_slot(u_addr, p_cached)
            raw_hex = self.get_storage_at(contract_address, slot_hash)
            try:
                raw_int = int(raw_hex, 16)
                if 0 < raw_int <= 10**27:
                    return float(raw_int) / 1e18
            except ValueError:
                pass

        # Probe base mapping slots p = 0 .. num_tracked_slots - 1
        for p in range(self.num_tracked_slots):
            slot_hash = compute_solidity_mapping_slot(u_addr, p)
            raw_hex = self.get_storage_at(contract_address, slot_hash)
            try:
                raw_int = int(raw_hex, 16)
                if 10**12 <= raw_int <= 10**27:
                    self._inferred_mapping_slot[addr_key] = p
                    return float(raw_int) / 1e18
            except ValueError:
                continue
        return 0.0

    # -------------------------------------------------------------------------
    # Unified 4-Tier Queries for Total Liability & Agent Stake
    # -------------------------------------------------------------------------
    def query_total_deposited(
        self,
        contract_address: str,
        extra_sigs: Optional[List[str]] = None,
        vault_eth_hint: float = 0.0,
    ) -> float:
        """
        Unified 4-Tier resolution for `vault_total_deposited`:
          1. Dynamic ABI Introspection getters (if registered) + extra_sigs
          2. Multi-Standard ABI Probing (ERC-4626, ERC-20, Vault, Staking)
          3. Raw EVM Storage Slot Inference (`0x0`..`0x7`)
        """
        addr_key = contract_address.lower()
        merged_sigs: List[str] = []
        for s in (
            (self._dynamic_global_getters.get(addr_key, []))
            + (extra_sigs or [])
            + self.STANDARD_GLOBAL_LIABILITY_SIGS
        ):
            if s not in merged_sigs:
                merged_sigs.append(s)

        for sig in merged_sigs:
            try:
                out, _, code = self._run_cast(["call", contract_address, sig])
                if code == 0 and out:
                    val = self.parse_uint256_ether(out)
                    if val > 0.0:
                        return val
            except Exception:
                continue

        # Tier 3: Storage slot inference fallback
        return self.infer_global_liability_from_storage(
            contract_address, vault_eth_hint=vault_eth_hint
        )

    def query_agent_stake(
        self,
        contract_address: str,
        agent_address: Optional[str] = None,
        extra_sigs: Optional[List[str]] = None,
        fallback_token_address: Optional[str] = None,
    ) -> float:
        """
        Unified 4-Tier resolution for `agent_vault_balance`:
          1. Dynamic ABI Introspection `(address) -> uint256` getters (if registered)
          2. Multi-Standard ABI Probing (`balances`, `balanceOf`, `stakes`, `deposits`, `shares`)
          3. Pure-Bytecode Keccak256 Mapping Slot Inference (`keccak256(abi.encode(agent, p))`)
          4. Optional companion token `balanceOf(agent)` fallback
        """
        u_addr = agent_address or self.agent_address
        addr_key = contract_address.lower()
        merged_sigs: List[str] = []
        for s in (
            (self._dynamic_user_getters.get(addr_key, []))
            + (extra_sigs or [])
            + self.STANDARD_USER_STAKE_SIGS
        ):
            if s not in merged_sigs:
                merged_sigs.append(s)

        for sig in merged_sigs:
            try:
                out, _, code = self._run_cast(["call", contract_address, sig, u_addr])
                if code == 0 and out:
                    val = self.parse_uint256_ether(out)
                    if val > 0.0:
                        return val
            except Exception:
                continue

        # Tier 3: Probe `keccak256(abi.encode(u_addr, p))` in contract storage directly
        inferred_val = self.infer_mapping_balance_from_storage(contract_address, u_addr)
        if inferred_val > 0.0:
            return inferred_val

        # Tier 4: Optional companion token balance check
        if fallback_token_address:
            tok_val = self.call_uint256_ether(
                fallback_token_address, "balanceOf(address)(uint256)", u_addr
            )
            if tok_val > 0.0:
                return tok_val

        return 0.0

    # -------------------------------------------------------------------------
    # Universal Regular User Verification (Verifikasi Agen Pengguna Biasa)
    # -------------------------------------------------------------------------
    def verify_not_owner(
        self,
        contract_address: str,
        owner_sig: str = "owner()(address)",
    ) -> Dict[str, str]:
        """
        Verify on-chain that `self.agent_address` is NOT the owner of `contract_address`.
        Gracefully handles contracts without a standard `owner()` getter.
        """
        candidate_sigs = [owner_sig, "owner()(address)", "admin()(address)", "governance()(address)"]
        contract_owner = "0x0000000000000000000000000000000000000000"
        for sig in candidate_sigs:
            owner_out, _, code = self._run_cast(["call", contract_address, sig])
            if code == 0 and owner_out.strip().startswith("0x"):
                contract_owner = owner_out.strip().split()[0]
                break

        if contract_owner.lower() == self.agent_address.lower():
            raise PermissionError(
                f"Privilege violation: Agent {self.agent_address} is the owner of {contract_address}!"
            )

        return {
            "agent_address": self.agent_address,
            "contract_address": contract_address,
            "contract_owner": contract_owner,
            "standing": "REGULAR_USER_EOA",
        }

    # -------------------------------------------------------------------------
    # Universal Read-Only Public Queries & Multi-ABI Fallbacks (Zero Gas)
    # -------------------------------------------------------------------------
    def get_block_number(self) -> int:
        out, _, code = self._run_cast(["block-number"])
        if code != 0 or not out:
            return 0
        try:
            return int(out.split()[0], 0)
        except ValueError:
            return 0

    def get_eth_balance(self, address: str) -> float:
        out, _, code = self._run_cast(["balance", address])
        if code != 0:
            return 0.0
        return self.parse_uint256_ether(out)

    def get_code_alive(self, address: str) -> bool:
        out, _, code = self._run_cast(["code", address])
        return code == 0 and len(out) > 2 and out != "0x"

    def get_storage_at(self, contract_address: str, slot: int | str) -> str:
        """Read a raw 32-byte storage slot via standard public `eth_getStorageAt`."""
        slot_hex = hex(slot) if isinstance(slot, int) else str(slot)
        res = self._json_rpc("eth_getStorageAt", [contract_address, slot_hex, "latest"])
        if isinstance(res, str) and res.startswith("0x"):
            return res.lower()
        out, _, code = self._run_cast(["storage", contract_address, slot_hex])
        if code == 0 and out:
            return out.split()[0].strip().lower()
        return "0x" + ("0" * 64)

    def call_uint256_ether(
        self, contract_address: str, sig: str, *call_args: str
    ) -> float:
        cmd = ["call", contract_address, sig] + list(call_args)
        out, _, code = self._run_cast(cmd)
        if code != 0:
            return 0.0
        return self.parse_uint256_ether(out)

    def call_uint256_ether_with_fallbacks(
        self,
        contract_address: str,
        candidate_sigs: List[str],
        *call_args: str,
        fallback_storage_slot: Optional[int] = None,
    ) -> float:
        """
        Try multiple ABI getter signatures in order; if all revert or return 0,
        fall back to Tier 3 Keccak256 mapping inference (if an address argument is passed)
        or raw storage slot inspection.
        """
        for sig in candidate_sigs:
            try:
                cmd = ["call", contract_address, sig] + list(call_args)
                out, _, code = self._run_cast(cmd)
                if code == 0 and out:
                    val = self.parse_uint256_ether(out)
                    if val > 0.0:
                        return val
            except Exception:
                continue

        # Tier 3a: If querying a user address balance, probe mapping(address => uint256) slots
        if len(call_args) == 1 and call_args[0].startswith("0x") and len(call_args[0]) == 42:
            mapped_val = self.infer_mapping_balance_from_storage(contract_address, call_args[0])
            if mapped_val > 0.0:
                return mapped_val

        # Tier 3b: If querying a global uint256, check fallback_storage_slot or scan base slots
        if fallback_storage_slot is not None:
            raw_hex = self.get_storage_at(contract_address, fallback_storage_slot)
            try:
                raw_int = int(raw_hex, 16)
                if 10**12 <= raw_int <= 10**27:
                    return float(raw_int) / 1e18
            except ValueError:
                pass
        return 0.0

    def call_bool(
        self, contract_address: str, sig: str, *call_args: str
    ) -> Tuple[bool, bool]:
        """Returns (call_succeeded, bool_value)."""
        cmd = ["call", contract_address, sig] + list(call_args)
        out, _, code = self._run_cast(cmd)
        if code != 0:
            return False, False
        return True, ("true" in out.lower())

    # -------------------------------------------------------------------------
    # Hybrid Greybox Opcode Trace & Public Storage Slot Diffing (x6 & x7)
    # -------------------------------------------------------------------------
    def snapshot_contract_storage(self, contract_address: str) -> Dict[str, str]:
        """
        Read current base storage slots (`0x0`..`0x7`) AND agent's `keccak256(abi.encode(agent, p))`
        mapping slots prior to transaction execution for black-box fallback diffing.
        """
        addr_key = contract_address.lower()
        snap: Dict[str, str] = {}
        for slot_idx in range(self.num_tracked_slots):
            slot_key = hex(slot_idx)
            snap[slot_key] = self.get_storage_at(contract_address, slot_key)
            map_key = compute_solidity_mapping_slot(self.agent_address, slot_idx)
            snap[map_key] = self.get_storage_at(contract_address, map_key)

        if addr_key not in self._observed_sload_slots:
            self._observed_sload_slots[addr_key] = set()

        self._last_storage_snapshot[addr_key] = snap
        return snap

    @staticmethod
    def _normalize_slot(slot_raw: str) -> str:
        try:
            return hex(int(slot_raw, 16))
        except ValueError:
            return slot_raw.lower()

    def trace_transaction_dataflow(
        self,
        tx_hash: Optional[str],
        contract_address: str,
        pre_storage: Optional[Dict[str, str]] = None,
    ) -> Tuple[float, float]:
        """
        Hybrid evaluation of EVM execution trace signals:
          - x6_sstore_execution_flag: 1.0 if transaction executed at least one SSTORE, else 0.0
          - x7_dataflow_waypoint_flag: 1.0 if SSTORE wrote to a slot previously read via SLOAD
            in prior transactions (SMARTIAN / ItyFuzz cross-tx dataflow waypoint), else 0.0

        Strategy:
          1. Greybox Primary: Query `debug_traceTransaction` (structLogs or prestateTracer).
          2. Black-Box Fallback: Diff public `eth_getStorageAt` slots (base + mapping slots).
        """
        addr_key = contract_address.lower()
        if addr_key not in self._observed_sload_slots:
            self._observed_sload_slots[addr_key] = set()

        # 1. Try Greybox structLogs Opcode Trace (SLOAD & SSTORE inspection)
        if tx_hash and tx_hash.startswith("0x") and len(tx_hash) == 66:
            trace_res = self._json_rpc(
                "debug_traceTransaction",
                [
                    tx_hash,
                    {"disableMemory": True, "disableStorage": False, "disableStack": False},
                ],
            )
            if isinstance(trace_res, dict) and isinstance(trace_res.get("structLogs"), list):
                struct_logs = trace_res["structLogs"]
                tx_sloads: Set[str] = set()
                tx_sstores: Set[str] = set()

                for step in struct_logs:
                    op = step.get("op")
                    stack = step.get("stack")
                    if op == "SLOAD" and stack and len(stack) >= 1:
                        slot_norm = self._normalize_slot(stack[-1])
                        tx_sloads.add(slot_norm)
                    elif op == "SSTORE" and stack and len(stack) >= 1:
                        slot_norm = self._normalize_slot(stack[-1])
                        tx_sstores.add(slot_norm)

                prior_sloads = self._observed_sload_slots[addr_key]
                has_sstore = 1.0 if len(tx_sstores) > 0 else 0.0
                has_waypoint = (
                    1.0
                    if len(tx_sstores.intersection(prior_sloads)) > 0
                    else 0.0
                )
                self._observed_sload_slots[addr_key].update(tx_sloads)
                return has_sstore, has_waypoint

            # 2. Try Greybox prestateTracer (read & write state diffs)
            diff_res = self._json_rpc(
                "debug_traceTransaction",
                [tx_hash, {"tracer": "prestateTracer", "tracerConfig": {"diffMode": True}}],
            )
            pre_res = self._json_rpc(
                "debug_traceTransaction",
                [tx_hash, {"tracer": "prestateTracer", "tracerConfig": {"diffMode": False}}],
            )
            if isinstance(diff_res, dict) and isinstance(pre_res, dict):
                tx_sloads = set()
                for acc, acc_data in pre_res.items():
                    if acc.lower() == addr_key and isinstance(acc_data, dict):
                        for k in acc_data.get("storage", {}).keys():
                            tx_sloads.add(self._normalize_slot(k))

                tx_sstores = set()
                post_map = diff_res.get("post", {})
                if isinstance(post_map, dict):
                    for acc, acc_data in post_map.items():
                        if acc.lower() == addr_key and isinstance(acc_data, dict):
                            for k in acc_data.get("storage", {}).keys():
                                tx_sstores.add(self._normalize_slot(k))

                prior_sloads = self._observed_sload_slots[addr_key]
                has_sstore = 1.0 if len(tx_sstores) > 0 else 0.0
                has_waypoint = (
                    1.0
                    if len(tx_sstores.intersection(prior_sloads)) > 0
                    else 0.0
                )
                self._observed_sload_slots[addr_key].update(tx_sloads)
                return has_sstore, has_waypoint

        # 3. Fallback: Public Black-Box Storage Slot Diffing (base slots + Keccak256 mapping slots)
        prev_snap = pre_storage or self._last_storage_snapshot.get(addr_key, {})
        curr_snap: Dict[str, str] = {}
        modified_slots: Set[str] = set()
        active_slots: Set[str] = set()

        slots_to_check = [hex(i) for i in range(self.num_tracked_slots)] + [
            compute_solidity_mapping_slot(self.agent_address, i)
            for i in range(self.num_tracked_slots)
        ]

        for slot_key in slots_to_check:
            val = self.get_storage_at(contract_address, slot_key)
            curr_snap[slot_key] = val
            norm_k = self._normalize_slot(slot_key)
            if int(val, 16) != 0:
                active_slots.add(norm_k)
            if slot_key in prev_snap and prev_snap[slot_key] != val:
                modified_slots.add(norm_k)

        prior_sloads = self._observed_sload_slots[addr_key]
        has_sstore = 1.0 if len(modified_slots) > 0 else 0.0
        has_waypoint = (
            1.0 if len(modified_slots.intersection(prior_sloads)) > 0 else 0.0
        )

        self._observed_sload_slots[addr_key].update(active_slots)
        self._last_storage_snapshot[addr_key] = curr_snap
        return has_sstore, has_waypoint

    def extract_tx_hash(self, raw_tx_output: str) -> Optional[str]:
        """Extract 32-byte transaction hash from `cast send` output or latest block."""
        if raw_tx_output:
            # 1. Match standalone `transactionHash   0x...` line from `cast send`
            for line in raw_tx_output.splitlines():
                stripped = line.strip()
                if stripped.startswith("transactionHash"):
                    m = self.TX_HASH_REGEX.search(stripped)
                    if m:
                        return m.group(0)
            # 2. Match JSON `"transactionHash":"0x..."`
            m_json = re.search(r'"transactionHash"\s*:\s*"(0x[a-fA-F0-9]{64})"', raw_tx_output)
            if m_json:
                return m_json.group(1)

        # Fallback: fetch latest block's last transaction hash via JSON-RPC
        block = self._json_rpc("eth_getBlockByNumber", ["latest", False])
        if isinstance(block, dict):
            txs = block.get("transactions", [])
            if isinstance(txs, list) and len(txs) > 0 and isinstance(txs[-1], str):
                return txs[-1]
        return None

    # -------------------------------------------------------------------------
    # Universal Signed EOA Transaction Execution
    # -------------------------------------------------------------------------
    def send_eoa_transaction(
        self,
        target_address: str,
        function_sig: str = "",
        call_args: Optional[List[str]] = None,
        eth_value: float = 0.0,
        sender_private_key: Optional[str] = None,
        gas_price_gwei: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Sign and broadcast a standard EOA transaction to Anvil.
        """
        pk = sender_private_key or self.agent_private_key
        cast_args: List[str] = ["send", target_address]

        if function_sig:
            cast_args.append(function_sig)
            if call_args:
                cast_args.extend(call_args)

        if eth_value > 0.0:
            wei_val = int(round(eth_value * 1e18))
            cast_args.extend(["--value", str(wei_val)])

        if gas_price_gwei is not None and gas_price_gwei > 0.0:
            wei_gas = int(round(gas_price_gwei * 1e9))
            cast_args.extend(["--gas-price", str(wei_gas)])

        cast_args.extend(["--private-key", pk])
        out, err, code = self._run_cast(cast_args)
        if code == 0:
            return True, out
        return False, err
