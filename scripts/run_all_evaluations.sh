#!/usr/bin/env bash
# run_all_evaluations.sh — Reproduce every result artifact from scratch.
#
# Usage (from project root):
#   bash scripts/run_all_evaluations.sh
#
# Prerequisites:
#   pip install -r requirements.txt
#   .env file with GROQ_API_KEY (optional — only needed for LLM report page)
#
# WARNING: tier_sensitivity.py and compare_rca_methods.py each take
# 20-60 minutes on CPU. The full pipeline runs for ~2 hours.

set -e   # abort on first error

echo "======================================================"
echo "  FULL EVALUATION PIPELINE — Thesis Industrial AI V2"
echo "======================================================"
echo ""

echo "[1/8] Data pipeline..."
python src/data_pipeline.py

echo ""
echo "[2/8] Training all 6 algorithms (RF last — becomes champion)..."
for algo in GB XGB MLP DT KNN RF; do
    echo "  -> $algo"
    python src/model_factory.py --algo $algo
done

echo ""
echo "[3/8] Test-set accuracy evaluation..."
python scripts/evaluate_test_set.py

echo ""
echo "[4/8] RCA batch — validation split..."
python scripts/evaluate_rca.py --split val

echo ""
echo "[5/8] RCA batch — test split..."
python scripts/evaluate_rca.py --split test

echo ""
echo "[6/8] Tier cascade sensitivity sweep (long — ~30 min)..."
python scripts/tier_sensitivity.py

echo ""
echo "[7/8] Greedy vs 3-tier comparison (long — ~20 min)..."
python scripts/compare_rca_methods.py

echo ""
echo "[8/8] PSI drift validation sweep..."
python scripts/drift_validation.py

echo ""
echo "======================================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "  See models/*.json and models/*.csv for results."
echo "======================================================"
