"""
evaluate_rca.py
Batch counterfactual RCA evaluation across all non-conforming samples
in either the validation or held-out test split.

Usage:
    python scripts/evaluate_rca.py --split val   # validation set (default)
    python scripts/evaluate_rca.py --split test  # held-out test set

Outputs are written to models/rca_evaluation_{split}.json and
models/rca_results_{split}.csv. Existing files are never overwritten.

Warning: this script trains the MLP validator and runs LIME for every
sample — expect 10–30 minutes depending on hardware.
"""

import argparse, os, sys, json, time
import numpy as np
import pandas as pd
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

from counterfactual_rca import CounterfactualRCA, TARGET_CLASSES

PROC = os.path.join(BASE, "data", "processed")
CKPT = os.path.join(BASE, "models", "checkpoints")

CLASS_NAMES = {0: "Waste", 1: "Acceptable", 2: "Target", 3: "Inefficient"}


def main(split: str = "val") -> None:
    out_json = os.path.join(BASE, "models", f"rca_evaluation_{split}.json")
    out_csv  = os.path.join(BASE, "models", f"rca_results_{split}.csv")

    if split == "val":
        x_file, y_file = "X_val.csv", "y_val.csv"
    elif split == "test":
        x_file, y_file = "X_test.csv", "y_test.csv"
    else:
        raise ValueError(f"--split must be 'val' or 'test', got '{split}'")

    print(f"=== RCA batch evaluation — split: {split} ===")
    print("Loading data and initialising RCA engine...")
    scaler        = joblib.load(os.path.join(CKPT, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(CKPT, "feature_names.pkl"))
    X_scaled      = pd.read_csv(os.path.join(PROC, x_file))
    y_raw         = pd.read_csv(os.path.join(PROC, y_file)).values.ravel()
    y             = y_raw - 1 if y_raw.min() > 0 else y_raw

    rca = CounterfactualRCA()

    nc_idx = np.where(~np.isin(y, list(TARGET_CLASSES)))[0]
    print(f"Non-conforming samples in {split} split: {len(nc_idx)}")

    rows = []
    t0 = time.time()

    for i, idx in enumerate(nc_idx):
        sample_scaled = X_scaled.iloc[[idx]]
        sample_real   = pd.DataFrame(
            scaler.inverse_transform(sample_scaled),
            columns=feature_names
        )
        try:
            result = rca.analyze(sample_real)
        except Exception as e:
            print(f"  WARNING: sample {idx} failed: {e}")
            result = {
                "tier": -1, "status": "error",
                "prediction": int(y[idx]),
                "confidence": 0.0,
                "adjustments": [],
                "validator_ok": False,
                "cf_confidence": 0.0,
                "message": str(e),
            }

        rows.append({
            "sample_idx":    int(idx),
            "true_class":    int(y[idx]),
            "true_label":    CLASS_NAMES[int(y[idx])],
            "pred_class":    result["prediction"],
            "input_conf":    round(result["confidence"], 4),
            "tier":          result["tier"],
            "status":        result["status"],
            "cf_confidence": result.get("cf_confidence", 0.0),
            "validator_ok":  result["validator_ok"],
            "n_adjustments": len(result["adjustments"]),
            "top_feature":   (result["adjustments"][0]["feature"]
                              if result["adjustments"] else ""),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(nc_idx):
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(nc_idx) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(nc_idx)}]  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    total      = len(df)
    tc         = df["tier"].value_counts().to_dict()
    t1         = int(tc.get(1, 0))
    t2         = int(tc.get(2, 0))
    t3         = int(tc.get(3, 0))
    err        = int(tc.get(-1, 0))
    resolved   = t1 + t2
    res_pct    = resolved / max(1, total)

    resolved_df    = df[df["tier"].isin([1, 2])]
    validated      = int(resolved_df["validator_ok"].sum())
    unvalidated    = len(resolved_df) - validated
    validation_pct = validated / max(1, len(resolved_df))

    top_features = (
        df[df["top_feature"] != ""]["top_feature"]
        .value_counts().head(5).to_dict()
    )

    summary = {
        "split": split,
        "total_non_conforming": total,
        "tier_counts": {
            "tier1_resolved":  t1,
            "tier2_resolved":  t2,
            "tier3_escalated": t3,
            "errors":          err,
        },
        "resolution_rate":       round(res_pct, 4),
        "tier1_pct":             round(t1 / max(1, total), 4),
        "tier2_pct":             round(t2 / max(1, total), 4),
        "tier3_pct":             round(t3 / max(1, total), 4),
        "validator_confirmed":   validated,
        "validator_unconfirmed": unvalidated,
        "validator_rate":        round(validation_pct, 4),
        "top_adjusted_features": top_features,
    }

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  RCA BATCH EVALUATION — {split.upper()} SPLIT")
    print("=" * 55)
    print(f"  Total non-conforming: {total}")
    print(f"  Tier 1 (SHAP+LIME):   {t1:4d}  ({t1/max(1,total):.1%})")
    print(f"  Tier 2 (NN-Anchored): {t2:4d}  ({t2/max(1,total):.1%})")
    print(f"  Tier 3 (Escalation):  {t3:4d}  ({t3/max(1,total):.1%})")
    if err:
        print(f"  Errors:               {err:4d}")
    print(f"\n  Overall resolution rate: {res_pct:.1%}")
    print(f"  Validator-confirmed:  {validated}/{resolved}  ({validation_pct:.1%})")
    print(f"  Validator-unconfirmed:{unvalidated}/{resolved}")
    print(f"\n  Top adjusted features:")
    for feat, count in top_features.items():
        print(f"    {feat}: {count} times")
    print(f"\n  Saved: {out_json}")
    print(f"  Saved: {out_csv}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch RCA evaluation")
    parser.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Which data split to evaluate (default: val)"
    )
    args = parser.parse_args()
    main(args.split)
