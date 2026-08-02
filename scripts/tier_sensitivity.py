"""
tier_sensitivity.py — Sweep confidence_threshold and max_iter to characterise
how sensitive the 3-tier cascade resolution rate is to these hyper-parameters.

Usage:
    python scripts/tier_sensitivity.py --split val   # default
    python scripts/tier_sensitivity.py --split test

Threshold sweep : {0.55, 0.65, 0.75, 0.85, 0.95} with max_iter fixed at 150.
MAX_ITER sweep  : {10, 25, 50, 150} with threshold fixed at 0.55.

Output files (never overwrite existing):
  models/tier_sensitivity_{split}.json
  models/tier_sensitivity_{split}.csv

Run time: ~20–120 min depending on hardware and split size.
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

THRESHOLD_SWEEP = [0.55, 0.65, 0.75, 0.85, 0.95]
MAX_ITER_SWEEP  = [10, 25, 50, 150]


def run_config(rca, X_scaled, y, scaler, feature_names):
    nc_idx = np.where(~np.isin(y, list(TARGET_CLASSES)))[0]
    t1 = t2 = t3 = err = validated = 0

    for idx in nc_idx:
        sample_scaled = X_scaled.iloc[[idx]]
        sample_real   = pd.DataFrame(
            scaler.inverse_transform(sample_scaled),
            columns=feature_names
        )
        try:
            result = rca.analyze(sample_real)
        except Exception:
            err += 1
            continue

        tier = result["tier"]
        if tier == 1:
            t1 += 1
            if result["validator_ok"]:
                validated += 1
        elif tier == 2:
            t2 += 1
            if result["validator_ok"]:
                validated += 1
        elif tier == 3:
            t3 += 1

    total    = len(nc_idx)
    resolved = t1 + t2
    res_pct  = resolved / max(1, total)
    val_rate = validated / max(1, resolved)
    return {
        "total": total, "tier1": t1, "tier2": t2, "tier3": t3, "errors": err,
        "resolved": resolved, "resolution_rate": round(res_pct, 4),
        "validator_confirmed": validated, "validator_rate": round(val_rate, 4),
    }


def main(split: str = "val") -> None:
    out_json = os.path.join(BASE, "models", f"tier_sensitivity_{split}.json")
    out_csv  = os.path.join(BASE, "models", f"tier_sensitivity_{split}.csv")

    if split == "val":
        x_file, y_file = "X_val.csv", "y_val.csv"
    elif split == "test":
        x_file, y_file = "X_test.csv", "y_test.csv"
    else:
        raise ValueError(f"--split must be 'val' or 'test', got '{split}'")

    print(f"=== Tier sensitivity sweep — split: {split} ===")
    print("Loading data...")
    scaler        = joblib.load(os.path.join(CKPT, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(CKPT, "feature_names.pkl"))
    X_scaled      = pd.read_csv(os.path.join(PROC, x_file))
    y_raw         = pd.read_csv(os.path.join(PROC, y_file)).values.ravel()
    y             = y_raw - 1 if y_raw.min() > 0 else y_raw

    records   = []
    t_global  = time.time()

    def _flush(total_so_far: float) -> None:
        """Write current records to disk after each completed config."""
        with open(out_json, "w") as f:
            json.dump({"split": split, "records": records,
                       "total_elapsed_s": round(total_so_far, 1),
                       "complete": False}, f, indent=2)
        pd.DataFrame(records).to_csv(out_csv, index=False)

    print(f"\n=== Part A: confidence_threshold sweep (max_iter=150, {split}) ===")
    for thresh in THRESHOLD_SWEEP:
        print(f"\n  threshold={thresh:.2f} ...", flush=True)
        rca = CounterfactualRCA(confidence_threshold=thresh, max_iter=150)
        t0  = time.time()
        stats = run_config(rca, X_scaled, y, scaler, feature_names)
        elapsed = time.time() - t0
        records.append({
            "split": split, "sweep": "threshold",
            "confidence_threshold": thresh, "max_iter": 150,
            **stats, "elapsed_s": round(elapsed, 1),
        })
        print(f"    resolution={stats['resolution_rate']:.1%}  "
              f"validator={stats['validator_rate']:.1%}  ({elapsed:.0f}s)")
        _flush(time.time() - t_global)

    print(f"\n=== Part B: max_iter sweep (threshold=0.55, {split}) ===")
    for mi in MAX_ITER_SWEEP:
        if mi == 150:
            continue  # already captured in Part A
        print(f"\n  max_iter={mi} ...", flush=True)
        rca = CounterfactualRCA(confidence_threshold=0.55, max_iter=mi)
        t0  = time.time()
        stats = run_config(rca, X_scaled, y, scaler, feature_names)
        elapsed = time.time() - t0
        records.append({
            "split": split, "sweep": "max_iter",
            "confidence_threshold": 0.55, "max_iter": mi,
            **stats, "elapsed_s": round(elapsed, 1),
        })
        print(f"    resolution={stats['resolution_rate']:.1%}  "
              f"validator={stats['validator_rate']:.1%}  ({elapsed:.0f}s)")
        _flush(time.time() - t_global)

    total_elapsed = time.time() - t_global

    with open(out_json, "w") as f:
        json.dump({"split": split, "records": records,
                   "total_elapsed_s": round(total_elapsed, 1),
                   "complete": True}, f, indent=2)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print(f"  SENSITIVITY SWEEP — {split.upper()} SPLIT")
    print("=" * 60)
    print(df[["sweep", "confidence_threshold", "max_iter",
              "resolution_rate", "validator_rate", "tier3"]].to_string(index=False))
    print(f"\n  Total elapsed: {total_elapsed/60:.1f} min")
    print(f"  Saved: {out_json}")
    print(f"  Saved: {out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tier cascade sensitivity sweep")
    parser.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Data split to evaluate (default: val)"
    )
    args = parser.parse_args()
    main(args.split)
