#!/usr/bin/env bash
# build_thesis_assets.sh — Regenerate the entire thesis_assets/ deliverable
# from scratch, in dependency order. Fails fast on the first error.
#
# Usage:
#   bash scripts/build_thesis_assets.sh
#
# Stage order matches the data dependencies between scripts:
#   1. export_dataset_stats.py    (no dependencies)
#   2. export_xai_results.py      (no dependencies)
#   3. export_rca_cases.py        (reads models/rca_results_test.csv)
#   4. export_quality_modules.py  (optionally reads Task 3's counterfactual_cases.json)
#   5. generate_thesis_figures.py (reads Task 1-4 outputs + models/*.json)
#   6. export_thesis_tables.py    (reads Task 1-3 outputs + models/*.json + docs/)

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

if [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python"
fi

echo "Using interpreter: $PYTHON"
echo "================================================================"

run_stage() {
    local name="$1"
    local script="$2"
    echo ""
    echo "--- Stage: $name ---"
    local t0=$SECONDS
    "$PYTHON" "$script"
    local elapsed=$((SECONDS - t0))
    echo "--- Stage '$name' finished in ${elapsed}s ---"
}

BUILD_START=$SECONDS

run_stage "1/6 Dataset statistics"   "scripts/export_dataset_stats.py"
run_stage "2/6 XAI batch export"     "scripts/export_xai_results.py"
run_stage "3/6 RCA worked cases"     "scripts/export_rca_cases.py"
run_stage "4/6 Quality 4.0 modules"  "scripts/export_quality_modules.py"
run_stage "5/6 Figure generation"    "scripts/generate_thesis_figures.py"
run_stage "6/6 Table export"         "scripts/export_thesis_tables.py"

TOTAL_ELAPSED=$((SECONDS - BUILD_START))
N_FILES=$(find thesis_assets -type f | wc -l | tr -d ' ')

echo ""
echo "================================================================"
echo "  BUILD COMPLETE"
echo "================================================================"
echo "  Total elapsed: ${TOTAL_ELAPSED}s"
echo "  Files under thesis_assets/: ${N_FILES}"
echo "================================================================"
