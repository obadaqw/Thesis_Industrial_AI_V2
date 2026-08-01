"""
drift_validation.py — Validate PSI detector sensitivity by sweeping synthetic
drift magnitudes k ∈ {0, 1, 2, 3} for each of the 13 sensor features.

For each (feature, k) combination:
  1. Copy the validation set.
  2. Shift the selected feature by k × σ_train, clip to [-1, 1].
  3. Compute PSI between training reference and the shifted validation column.
  4. Record PSI and its status (stable / moderate / critical).

Results saved to:
  models/drift_validation.json
  models/drift_validation.csv

This table appears in the thesis as evidence that the PSI detector reliably
triggers at the expected drift magnitudes, validating its use for production
monitoring.
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

from drift_detector import DriftDetector, psi_status

PROC = os.path.join(BASE, "data", "processed")
CKPT = os.path.join(BASE, "models", "checkpoints")

K_VALUES = [0, 1, 2, 3]


def main():
    print("Loading data...")
    feature_names = joblib.load(os.path.join(CKPT, "feature_names.pkl"))
    X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
    X_val   = pd.read_csv(os.path.join(PROC, "X_val.csv"))

    detector = DriftDetector(X_train, feature_names)

    records = []
    for feat in feature_names:
        sigma = float(X_train[feat].std())
        for k in K_VALUES:
            X_drifted = X_val.copy()
            X_drifted[feat] = (X_drifted[feat] + k * sigma).clip(-1.0, 1.0)
            psi = detector.feature_psi(feat, X_drifted)
            status = psi_status(psi)
            records.append({
                "feature": feat,
                "k_sigma": k,
                "sigma":   round(sigma, 6),
                "shift":   round(k * sigma, 6),
                "psi":     psi,
                "status":  status,
            })
            print(f"  {feat:45s}  k={k}  PSI={psi:.4f}  [{status}]")

    df = pd.DataFrame(records)

    out_json = os.path.join(BASE, "models", "drift_validation.json")
    out_csv  = os.path.join(BASE, "models", "drift_validation.csv")

    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    df.to_csv(out_csv, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PSI SENSITIVITY SUMMARY  (counts by k × status)")
    print("=" * 70)
    pivot = df.groupby(["k_sigma", "status"]).size().unstack(fill_value=0)
    print(pivot.to_string())
    print(f"\n  Saved: {out_json}")
    print(f"  Saved: {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
