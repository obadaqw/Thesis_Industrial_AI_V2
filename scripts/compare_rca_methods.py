"""
compare_rca_methods.py — Compare 3-tier CounterfactualRCA vs greedy baseline
on the validation set.

Metrics computed for each method:
  resolution_rate   — % of non-conforming samples where a CF is found
  validator_rate    — % of resolved cases confirmed by MLP validator
  mean_proximity    — mean L2(original_scaled, cf_scaled) for resolved cases
                      (lower = counterfactual closer to original = more realistic)
  mean_sparsity     — mean number of features with |delta|>1e-4 in CF
                      (lower = fewer parameter changes = more actionable)
  plausibility_rate — % of CF feature values within training data min/max bounds
                      (should be ~100% since we clip to [-1,1])

Outputs:
  models/rca_comparison.json
  models/rca_comparison.csv
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

from counterfactual_rca import CounterfactualRCA, TARGET_CLASSES
from greedy_rca import GreedyCounterfactual

PROC = os.path.join(BASE, "data", "processed")
CKPT = os.path.join(BASE, "models", "checkpoints")


def compute_proximity(orig_scaled, cf_scaled):
    return float(np.linalg.norm(orig_scaled - cf_scaled))


def compute_sparsity(adjustments):
    return sum(1 for a in adjustments if abs(a["delta"]) > 1e-4)


def compute_plausibility(cf_scaled, X_train):
    """% of CF feature values within training-data column ranges."""
    lo  = X_train.min(axis=0)
    hi  = X_train.max(axis=0)
    in_bounds = np.logical_and(cf_scaled >= lo, cf_scaled <= hi)
    return float(in_bounds.mean())


def evaluate(method_name, engine, X_scaled, y, scaler, feature_names, X_train):
    nc_idx = np.where(~np.isin(y, list(TARGET_CLASSES)))[0]
    rows   = []
    t0     = time.time()

    for i, idx in enumerate(nc_idx):
        sample_scaled = X_scaled.iloc[[idx]]
        sample_real   = pd.DataFrame(
            scaler.inverse_transform(sample_scaled), columns=feature_names
        )
        x_orig = sample_scaled.values[0]

        try:
            result = engine.analyze(sample_real)
        except Exception as e:
            rows.append({
                "method": method_name, "sample_idx": int(idx),
                "resolved": False, "validator_ok": False,
                "proximity": np.nan, "sparsity": np.nan, "plausibility": np.nan,
                "tier": -1,
            })
            continue

        resolved = result["status"] == "resolved"
        tier     = result["tier"]
        vok      = result.get("validator_ok", False)

        prox  = np.nan
        spar  = np.nan
        plaus = np.nan

        if resolved and result.get("adjustments"):
            cf_real = np.array([
                [a["suggested"] for a in sorted(result["adjustments"],
                                                key=lambda a: feature_names.index(a["feature"]))]
            ])
            # Rebuild full CF in scaled space via scaler
            # Reconstruct from adjustments: start from original, apply deltas
            cf_real_full = scaler.inverse_transform(sample_scaled.values)[0].copy()
            for a in result["adjustments"]:
                fi = feature_names.index(a["feature"])
                cf_real_full[fi] = a["suggested"]
            cf_scaled = scaler.transform([cf_real_full])[0]

            prox  = compute_proximity(x_orig, cf_scaled)
            spar  = compute_sparsity(result["adjustments"])
            plaus = compute_plausibility(cf_scaled, X_train)

        rows.append({
            "method":      method_name,
            "sample_idx":  int(idx),
            "resolved":    resolved,
            "validator_ok": bool(vok),
            "proximity":   prox,
            "sparsity":    spar,
            "plausibility": plaus,
            "tier":        tier,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(nc_idx):
            print(f"  [{i+1}/{len(nc_idx)}]  {time.time()-t0:.0f}s")

    return pd.DataFrame(rows)


def aggregate(df):
    total    = len(df)
    resolved = df[df["resolved"]]
    return {
        "total":           total,
        "resolved":        len(resolved),
        "resolution_rate": round(len(resolved) / max(1, total), 4),
        "validator_confirmed": int(resolved["validator_ok"].sum()),
        "validator_rate":  round(resolved["validator_ok"].mean() if len(resolved) else 0, 4),
        "mean_proximity":  round(resolved["proximity"].mean(), 4) if len(resolved) else np.nan,
        "mean_sparsity":   round(resolved["sparsity"].mean(), 4) if len(resolved) else np.nan,
        "plausibility_rate": round(resolved["plausibility"].mean(), 4) if len(resolved) else np.nan,
    }


def main():
    print("Loading data...")
    scaler        = joblib.load(os.path.join(CKPT, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(CKPT, "feature_names.pkl"))
    X_scaled      = pd.read_csv(os.path.join(PROC, "X_val.csv"))
    y_raw         = pd.read_csv(os.path.join(PROC, "y_val.csv")).values.ravel()
    y             = y_raw - 1 if y_raw.min() > 0 else y_raw
    X_train       = pd.read_csv(os.path.join(PROC, "X_train.csv")).values

    print("\n=== Evaluating 3-tier CounterfactualRCA ===")
    rca     = CounterfactualRCA()
    df_rca  = evaluate("3-tier RCA", rca, X_scaled, y, scaler, feature_names, X_train)
    agg_rca = aggregate(df_rca)

    print("\n=== Evaluating GreedyCounterfactual ===")
    greedy     = GreedyCounterfactual()
    df_greedy  = evaluate("Greedy", greedy, X_scaled, y, scaler, feature_names, X_train)
    agg_greedy = aggregate(df_greedy)

    # ── Save ──────────────────────────────────────────────────────────────────
    comparison = {
        "3-tier RCA": agg_rca,
        "Greedy":     agg_greedy,
    }

    out_json = os.path.join(BASE, "models", "rca_comparison.json")
    out_csv  = os.path.join(BASE, "models", "rca_comparison.csv")

    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2)

    df_all = pd.concat([df_rca, df_greedy], ignore_index=True)
    df_all.to_csv(out_csv, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  METHOD COMPARISON — VALIDATION SET")
    print("=" * 65)
    headers = ["Metric", "3-tier RCA", "Greedy"]
    metrics = [
        ("Resolution rate",   f"{agg_rca['resolution_rate']:.1%}",
                              f"{agg_greedy['resolution_rate']:.1%}"),
        ("Validator rate",    f"{agg_rca['validator_rate']:.1%}",
                              f"{agg_greedy['validator_rate']:.1%}"),
        ("Mean proximity",    f"{agg_rca['mean_proximity']:.4f}",
                              f"{agg_greedy['mean_proximity']:.4f}"),
        ("Mean sparsity",     f"{agg_rca['mean_sparsity']:.1f} features",
                              f"{agg_greedy['mean_sparsity']:.1f} features"),
        ("Plausibility rate", f"{agg_rca['plausibility_rate']:.1%}",
                              f"{agg_greedy['plausibility_rate']:.1%}"),
    ]
    print(f"  {'Metric':<22} {'3-tier RCA':>14} {'Greedy':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*14}")
    for m, r, g in metrics:
        print(f"  {m:<22} {r:>14} {g:>14}")
    print(f"\n  Saved: {out_json}")
    print(f"  Saved: {out_csv}")
    print("=" * 65)


if __name__ == "__main__":
    main()
