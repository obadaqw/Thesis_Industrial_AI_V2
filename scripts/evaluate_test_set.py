"""
evaluate_test_set.py
Final held-out test-set evaluation for the thesis.
Run ONCE after all design decisions are locked.
Outputs a classification report and saves results to
models/test_evaluation.json for use in Chapter 4.
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score
)

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC  = os.path.join(BASE, "data", "processed")
CKPT  = os.path.join(BASE, "models", "checkpoints")
OUT   = os.path.join(BASE, "models", "test_evaluation.json")

CLASS_NAMES = ["Waste", "Acceptable", "Target", "Inefficient"]

def main():
    model  = joblib.load(os.path.join(CKPT, "current_model.pkl"))
    X_test = pd.read_csv(os.path.join(PROC, "X_test.csv"))
    y_raw  = pd.read_csv(os.path.join(PROC, "y_test.csv")).values.ravel()
    y_test = y_raw - 1 if y_raw.min() > 0 else y_raw

    y_pred = model.predict(X_test)

    acc   = float(accuracy_score(y_test, y_pred))
    f1_w  = float(f1_score(y_test, y_pred, average="weighted"))
    f1_m  = float(f1_score(y_test, y_pred, average="macro"))
    cm    = confusion_matrix(y_test, y_pred).tolist()

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = y_test == i
        per_class[name] = {
            "precision": float(precision_score(
                y_test, y_pred, labels=[i], average="micro",
                zero_division=0)),
            "recall":    float(recall_score(
                y_test, y_pred, labels=[i], average="micro",
                zero_division=0)),
            "n_samples": int(mask.sum()),
        }

    # Non-conformance group metrics (Waste=0, Inefficient=3)
    nc_true = np.isin(y_test, [0, 3])
    nc_pred = np.isin(y_pred, [0, 3])
    tp_nc   = int(np.sum(nc_true & nc_pred))
    nc_precision = tp_nc / max(1, int(nc_pred.sum()))
    nc_recall    = tp_nc / max(1, int(nc_true.sum()))

    results = {
        "model": type(model).__name__,
        "n_samples": len(y_test),
        "accuracy":  round(acc,  4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro":    round(f1_m, 4),
        "per_class":   per_class,
        "non_conformance_precision": round(nc_precision, 4),
        "non_conformance_recall":    round(nc_recall, 4),
        "confusion_matrix": cm,
    }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*55)
    print("  FINAL TEST-SET EVALUATION")
    print("="*55)
    print(f"  Model:        {results['model']}")
    print(f"  n_samples:    {results['n_samples']}")
    print(f"  Accuracy:     {acc:.4f}  ({acc:.2%})")
    print(f"  F1 (weighted):{f1_w:.4f}")
    print(f"  F1 (macro):   {f1_m:.4f}")
    print("\n  Per-class Recall:")
    for name, v in per_class.items():
        print(f"    {name:12s}  recall={v['recall']:.4f}  "
              f"precision={v['precision']:.4f}  n={v['n_samples']}")
    print(f"\n  Non-conformance detection:")
    print(f"    Precision: {nc_precision:.4f}")
    print(f"    Recall:    {nc_recall:.4f}")
    print(f"\n  Saved to: {OUT}")
    print("="*55)
    print("\n  ⚠️  Use these numbers in Chapter 4 Table 4.1.")
    print("  Do NOT use validation-set numbers as final results.")

if __name__ == "__main__":
    main()
