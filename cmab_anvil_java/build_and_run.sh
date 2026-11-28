#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/target/classes"
TEST_OUT_DIR="${SCRIPT_DIR}/target/test-classes"

mkdir -p "${OUT_DIR}" "${TEST_OUT_DIR}"

echo "[*] Mengompilasi sumber kode utama Java FuzzCMAB..."
find "${SCRIPT_DIR}/src/main/java" -name "*.java" > "${SCRIPT_DIR}/target/main_sources.txt"
javac -d "${OUT_DIR}" @"${SCRIPT_DIR}/target/main_sources.txt"

echo "[*] Mengompilasi suite pengujian unit Java FuzzCMAB..."
find "${SCRIPT_DIR}/src/test/java" -name "*.java" > "${SCRIPT_DIR}/target/test_sources.txt"
javac -cp "${OUT_DIR}" -d "${TEST_OUT_DIR}" @"${SCRIPT_DIR}/target/test_sources.txt"

MODE="${1:-test}"
shift || true

if [[ "${MODE}" == "test" ]]; then
    echo "[*] Menjalankan FuzzCMABTestSuite..."
    java -ea -cp "${OUT_DIR}:${TEST_OUT_DIR}" id.ac.skripsi.fuzzcmab.FuzzCMABTestSuite "$@"
elif [[ "${MODE}" == "sim" ]]; then
    echo "[*] Menjalankan SimulationRunner (Agen_Vault_Custody)..."
    java -cp "${OUT_DIR}" id.ac.skripsi.fuzzcmab.experiments.SimulationRunner "$@"
elif [[ "${MODE}" == "matrix" ]]; then
    echo "[*] Menjalankan SkripsiExperimentRunner (Matriks Faktorial)..."
    java -cp "${OUT_DIR}" id.ac.skripsi.fuzzcmab.experiments.SkripsiExperimentRunner "$@"
else
    echo "Penggunaan: ./build_and_run.sh [test | sim | matrix] [args...]"
    exit 1
fi
