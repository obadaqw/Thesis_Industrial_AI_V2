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
# 20-120 minutes on CPU. The full pipeline runs for ~3-4 hours.

set -e   # abort on first error

echo "======================================================"
echo "  FULL EVALUATION PIPELINE — Thesis Industrial AI V2"
echo "======================================================"
echo ""

echo "[1/10] Data pipeline..."
python src/data_pipeline.py

echo ""
echo "[2/10] Training all 6 algorithms (RF last — becomes champion)..."
for algo in GB XGB MLP DT KNN RF; do
    echo "  -> $algo"
    python src/model_factory.py --algo $algo
done

echo ""
echo "[3/10] Tree depth empirical check..."
python scripts/check_tree_depth.py

echo ""
echo "[4/10] Test-set accuracy evaluation..."
python scripts/evaluate_test_set.py

echo ""
echo "[5/10] RCA batch — validation split..."
python scripts/evaluate_rca.py --split val

echo ""
echo "[6/10] RCA batch — test split..."
python scripts/evaluate_rca.py --split test

echo ""
echo "[7/10] Tier cascade sensitivity sweep — validation split (long — ~60 min)..."
python scripts/tier_sensitivity.py --split val

echo ""
echo "[8/10] Tier cascade sensitivity sweep — test split (long — ~60 min)..."
python scripts/tier_sensitivity.py --split test

echo ""
echo "[9/10] Greedy vs 3-tier comparison — validation split (long — ~20 min)..."
python scripts/compare_rca_methods.py --split val

echo ""
echo "[10/10] Greedy vs 3-tier comparison — test split (long — ~20 min)..."
python scripts/compare_rca_methods.py --split test

echo ""
echo "[Bonus] PSI drift validation sweep..."
python scripts/drift_validation.py

echo ""
echo "======================================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "  See models/*.json and models/*.csv for results."
echo "======================================================"
