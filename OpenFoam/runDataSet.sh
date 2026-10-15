#!/usr/bin/env bash
# Existing serial dataset workflow. Run from anywhere; output stays beside this file.
set -euo pipefail
BASE_CASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$BASE_CASE")"
PYTHON="${PYTHON:-python3}"
RUNS=$(cd "$PROJECT_ROOT" && "$PYTHON" -c 'from shred_utils.input_var import DATASET_CASES; print(DATASET_CASES)')
if [[ "${1:-}" == "--dry-run" ]]; then
    echo "Generate ${RUNS} cases using shred_utils/input_var.py."
    echo "Output: $BASE_CASE/case_omega* and $BASE_CASE/CSV_files"
    echo "Per case: test2.py -> blockMesh -> pimpleFoam -> foamToVTK; then pvpython DataSet_to_CSV.py"
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash OpenFoam/runDataSet.sh [--dry-run]" >&2; exit 2
fi
if (( RUNS < 1 )); then
    echo "Set positive dataset family counts in shred_utils/input_var.py." >&2; exit 2
fi
# Refuse a previous or partially generated campaign before touching any inputs.
if [[ -e "$BASE_CASE/CSV_files" ]] || compgen -G "$BASE_CASE/case_omega*" > /dev/null; then
    echo "Dataset output already exists. Move it aside or use a fresh copy of this repository." >&2
    exit 2
fi
for tool in blockMesh pimpleFoam foamToVTK pvpython; do
    command -v "$tool" >/dev/null || { echo "Missing $tool; load OpenFOAM/ParaView first." >&2; exit 2; }
done
cd "$BASE_CASE"
"$PYTHON" import_input.py
mkdir "$BASE_CASE/CSV_files"
cp "$PROJECT_ROOT/shred_utils/input_var.py" "$BASE_CASE/CSV_files/input_var.py"
for (( run=0; run<RUNS; run++ )); do
    CASE_DIR="$BASE_CASE/case_omega${run}"
    CSV_CASE_DIR="$BASE_CASE/CSV_files/case_omega${run}"
    mkdir "$CASE_DIR" "$CSV_CASE_DIR"
    cp -a "$BASE_CASE/constant" "$BASE_CASE/system" "$BASE_CASE/0" "$CASE_DIR/"
    "$PYTHON" "$PROJECT_ROOT/test2.py" "$run" --output "$CASE_DIR/0/omega_t_table.txt" > /dev/null
    cp "$CASE_DIR/0/omega_t_table.txt" "$CSV_CASE_DIR/"
    cd "$CASE_DIR"
    echo "Generating case ${run} of ${RUNS}"
    blockMesh > log.blockMesh 2>&1
    pimpleFoam > log.pimpleFoam 2>&1
    for timestep in [0-9]*; do
        if [[ -d "$timestep" ]]; then
            gunzip -f "$timestep"/*.gz 2>/dev/null || true
        fi
    done
    touch "case_omega${run}.foam"
    foamToVTK > log.foamToVTK 2>&1
    # Same post-export cleanup as the original launcher, inside this new case only.
    rm -rf -- [0-9]*
    cd "$BASE_CASE"
done
cd "$PROJECT_ROOT"
pvpython DataSet_to_CSV.py
