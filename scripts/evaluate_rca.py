"""
evaluate_rca.py
Batch counterfactual RCA evaluation across all non-conforming
validation samples. Run after design is locked.
Saves results to models/rca_evaluation.json and
models/rca_results.csv for use in Chapter 4 Tables 4.3 and 4.4.

Warning: this script trains the MLP validator and runs LIME for
every sample — expect 10-30 minutes depending on hardware.
Progress is printed every 10 samples.
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

from counterfactual_rca import CounterfactualRCA, TARGET_CLASSES

PROC  = os.path.join(BASE, "data", "processed")
CKPT  = os.path.join(BASE, "models", "checkpoints")
OUT_J = os.path.join(BASE, "models", "rca_evaluation.json")
OUT_C = os.path.join(BASE, "models", "rca_results.csv")

CLASS_NAMES = {0:"Waste", 1:"Acceptable", 2:"Target", 3:"Inefficient"}

def main():
    print("Loading data and initializing RCA engine...")
    scaler       = joblib.load(os.path.join(CKPT, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(CKPT, "feature_names.pkl"))
    X_val_scaled  = pd.read_csv(os.path.join(PROC, "X_val.csv"))
    y_raw         = pd.read_csv(os.path.join(PROC, "y_val.csv")).values.ravel()
    y_val         = y_raw - 1 if y_raw.min() > 0 else y_raw

    rca = CounterfactualRCA()

    # Only analyze non-conforming samples (Waste=0, Inefficient=3)
    nc_idx = np.where(~np.isin(y_val, list(TARGET_CLASSES)))[0]
    print(f"Non-conforming samples to analyze: {len(nc_idx)}")

    rows = []
    t0 = time.time()

    for i, idx in enumerate(nc_idx):
        sample_scaled = X_val_scaled.iloc[[idx]]
        sample_real   = pd.DataFrame(
            scaler.inverse_transform(sample_scaled),
            columns=feature_names
        )
        try:
            result = rca.analyze(sample_real)
        except Exception as e:
            print(f"  ⚠️  Sample {idx} failed: {e}")
            result = {
                "tier": -1, "status": "error",
                "prediction": int(y_val[idx]),
                "confidence": 0.0,
                "adjustments": [],
                "validator_ok": False,
                "cf_confidence": 0.0,
                "message": str(e),
            }

        rows.append({
            "sample_idx":     int(idx),
            "true_class":     int(y_val[idx]),
            "true_label":     CLASS_NAMES[int(y_val[idx])],
            "pred_class":     result["prediction"],
            "input_conf":     round(result["confidence"], 4),
            "tier":           result["tier"],
            "status":         result["status"],
            "cf_confidence":  result.get("cf_confidence", 0.0),
            "validator_ok":   result["validator_ok"],
            "n_adjustments":  len(result["adjustments"]),
            "top_feature":    (result["adjustments"][0]["feature"]
                               if result["adjustments"] else ""),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(nc_idx):
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(nc_idx) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(nc_idx)}]  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_C, index=False)

    # ── Aggregate statistics ──────────────────────────────────────
    total = len(df)
    tier_counts = df["tier"].value_counts().to_dict()

    t1 = int((tier_counts.get(1, 0)))
    t2 = int((tier_counts.get(2, 0)))
    t3 = int((tier_counts.get(3, 0)))
    err = int((tier_counts.get(-1, 0)))

    resolved       = t1 + t2
    resolved_pct   = resolved / max(1, total)

    # Validator stats (only on resolved samples)
    resolved_df    = df[df["tier"].isin([1, 2])]
    validated      = int(resolved_df["validator_ok"].sum())
    unvalidated    = len(resolved_df) - validated
    validation_pct = validated / max(1, len(resolved_df))

    # Top adjusted features across resolved samples
    top_features = (
        df[df["top_feature"] != ""]["top_feature"]
        .value_counts()
        .head(5)
        .to_dict()
    )

    summary = {
        "total_non_conforming": total,
        "tier_counts": {
            "tier1_resolved":  t1,
            "tier2_resolved":  t2,
            "tier3_escalated": t3,
            "errors":          err,
        },
        "resolution_rate": round(resolved_pct, 4),
        "tier1_pct": round(t1 / max(1, total), 4),
        "tier2_pct": round(t2 / max(1, total), 4),
        "tier3_pct": round(t3 / max(1, total), 4),
        "validator_confirmed":   validated,
        "validator_unconfirmed": unvalidated,
        "validator_rate": round(validation_pct, 4),
        "top_adjusted_features": top_features,
    }

    with open(OUT_J, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print Chapter 4 ready summary ─────────────────────────────
    print("\n" + "="*55)
    print("  RCA BATCH EVALUATION — CHAPTER 4 READY SUMMARY")
    print("="*55)
    print(f"  Total non-conforming samples: {total}")
    print(f"  Tier 1 (SHAP+LIME):  {t1:4d}  ({t1/max(1,total):.1%})")
    print(f"  Tier 2 (NN-Anchored):{t2:4d}  ({t2/max(1,total):.1%})")
    print(f"  Tier 3 (Escalation): {t3:4d}  ({t3/max(1,total):.1%})")
    if err:
        print(f"  Errors:              {err:4d}  (investigate!)")
    print(f"\n  Overall resolution rate: {resolved_pct:.1%}")
    print(f"  (Tier1 + Tier2 combined)")
    print(f"\n  Validator-confirmed: {validated}/{resolved}  "
          f"({validation_pct:.1%} of resolved)")
    print(f"  Validator-unconfirmed: {unvalidated}/{resolved}")
    print(f"\n  Top adjusted features:")
    for feat, count in top_features.items():
        print(f"    {feat}: {count} times")
    print(f"\n  Results saved to:")
    print(f"    {OUT_J}")
    print(f"    {OUT_C}")
    print("="*55)
    print("\n  ✅ Use summary for Chapter 4 Tables 4.3 and 4.4.")
    print("  ✅ Use rca_results.csv for per-case analysis.")
    print("  ⚠️  Warn in thesis: batch uses VALIDATION set,")
    print("      not the held-out test set.")

if __name__ == "__main__":
    main()
