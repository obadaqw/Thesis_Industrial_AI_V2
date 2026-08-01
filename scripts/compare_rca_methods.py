"""
compare_rca_methods.py — Compare 3-tier CounterfactualRCA against the
centroid-ablation baseline (GreedyCounterfactual) on a specified data split.

Usage:
    python scripts/compare_rca_methods.py --split val   # default
    python scripts/compare_rca_methods.py --split test

The centroid-ablation baseline applies the same acceptance criterion
(P(Acceptable) + P(Target) >= 0.55) as the 3-tier engine. Matched acceptance
criteria are a design requirement so that resolution-rate differences reflect the
effect of SHAP feature selection and LIME directional guidance alone.

Leakage guarantee: the MLP validator, the Tier-2 good-sample pool, and the greedy
centroid are all derived from X_train exclusively. The specified split is used
only as the evaluation target, not for any training step.

Metrics computed for each method:
  resolution_rate  — fraction of non-conforming samples where a CF is found
  validator_rate   — fraction of resolved cases confirmed by MLP validator (seed=7)
  mean_proximity   — mean L2(original_scaled, cf_scaled) for resolved cases
                     (lower = counterfactual is closer to the original cycle)
  mean_sparsity    — mean number of features with |delta| > 1e-4
                     (lower = fewer parameter changes = more actionable recipe)
  mean_nn_distance — mean L2 distance from the CF to its nearest conforming
                     training sample in scaled space
                     (lower = CF lies closer to the real conforming manifold)

McNemar's test (exact, two-sided) on paired per-sample validator outcomes
determines whether the validator-rate difference is distinguishable from chance.

Outputs (never overwrite existing files):
  models/rca_comparison_{split}.json
  models/rca_comparison_{split}.csv
"""

import argparse, os, sys, json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import euclidean_distances
from statsmodels.stats.contingency_tables import mcnemar

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

from counterfactual_rca import CounterfactualRCA, TARGET_CLASSES
from greedy_rca import GreedyCounterfactual

PROC = os.path.join(BASE, "data", "processed")
CKPT = os.path.join(BASE, "models", "checkpoints")


def compute_proximity(orig_scaled: np.ndarray, cf_scaled: np.ndarray) -> float:
    return float(np.linalg.norm(orig_scaled - cf_scaled))


def compute_sparsity(adjustments: list) -> int:
    return sum(1 for a in adjustments if abs(a["delta"]) > 1e-4)


def compute_nn_distance(cf_scaled: np.ndarray, good_samples: np.ndarray) -> float:
    """L2 distance from cf_scaled to its nearest conforming training sample."""
    dists = euclidean_distances([cf_scaled], good_samples)
    return float(dists.min())


def evaluate(method_name, engine, X_scaled, y, scaler, feature_names,
             X_train, good_samples):
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
        except Exception:
            rows.append({
                "method": method_name, "sample_idx": int(idx),
                "resolved": False, "validator_ok": False,
                "proximity": np.nan, "sparsity": np.nan,
                "nn_distance": np.nan, "tier": -1,
            })
            continue

        resolved = result["status"] == "resolved"
        tier     = result["tier"]
        vok      = result.get("validator_ok", False)

        prox = spar = nn_dist = np.nan

        if resolved and result.get("adjustments"):
            cf_real_full = scaler.inverse_transform(sample_scaled.values)[0].copy()
            for a in result["adjustments"]:
                fi = feature_names.index(a["feature"])
                cf_real_full[fi] = a["suggested"]
            cf_scaled_arr = scaler.transform([cf_real_full])[0]

            prox    = compute_proximity(x_orig, cf_scaled_arr)
            spar    = compute_sparsity(result["adjustments"])
            nn_dist = compute_nn_distance(cf_scaled_arr, good_samples)

        rows.append({
            "method":       method_name,
            "sample_idx":   int(idx),
            "resolved":     resolved,
            "validator_ok": bool(vok),
            "proximity":    prox,
            "sparsity":     spar,
            "nn_distance":  nn_dist,
            "tier":         tier,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(nc_idx):
            print(f"  [{i+1}/{len(nc_idx)}]  {time.time()-t0:.0f}s")

    return pd.DataFrame(rows)


def aggregate(df):
    total    = len(df)
    resolved = df[df["resolved"]]
    return {
        "total":                 total,
        "resolved":              len(resolved),
        "resolution_rate":       round(len(resolved) / max(1, total), 4),
        "validator_confirmed":   int(resolved["validator_ok"].sum()),
        "validator_rate":        round(resolved["validator_ok"].mean()
                                       if len(resolved) else 0.0, 4),
        "mean_proximity":        round(resolved["proximity"].mean(), 4)
                                       if len(resolved) else float("nan"),
        "mean_sparsity":         round(resolved["sparsity"].mean(), 4)
                                       if len(resolved) else float("nan"),
        "mean_nn_distance":      round(resolved["nn_distance"].mean(), 4)
                                       if len(resolved) else float("nan"),
    }


def run_mcnemar(df_rca, df_greedy):
    """McNemar's exact test on paired per-sample validator outcomes."""
    merged = df_rca[["sample_idx", "validator_ok"]].merge(
        df_greedy[["sample_idx", "validator_ok"]],
        on="sample_idx", suffixes=("_rca", "_greedy")
    )
    a = int(( merged["validator_ok_rca"] &  merged["validator_ok_greedy"]).sum())
    b = int((~merged["validator_ok_rca"] &  merged["validator_ok_greedy"]).sum())
    c = int(( merged["validator_ok_rca"] & ~merged["validator_ok_greedy"]).sum())
    d = int((~merged["validator_ok_rca"] & ~merged["validator_ok_greedy"]).sum())
    table = np.array([[a, b], [c, d]])
    result = mcnemar(table, exact=True)
    return {
        "contingency": {"a_both": a, "b_greedy_only": b,
                        "c_rca_only": c, "d_neither": d},
        "mcnemar_statistic": float(result.statistic),
        "mcnemar_p": float(result.pvalue),
        "interpretation": (
            "p >= 0.05: validator-rate difference not statistically significant"
            if result.pvalue >= 0.05 else
            "p < 0.05: validator-rate difference is statistically significant"
        ),
    }


def main(split: str = "val") -> None:
    out_json = os.path.join(BASE, "models", f"rca_comparison_{split}.json")
    out_csv  = os.path.join(BASE, "models", f"rca_comparison_{split}.csv")

    if split == "val":
        x_file, y_file = "X_val.csv", "y_val.csv"
    elif split == "test":
        x_file, y_file = "X_test.csv", "y_test.csv"
    else:
        raise ValueError(f"--split must be 'val' or 'test', got '{split}'")

    print(f"=== RCA method comparison — split: {split} ===")
    print("Loading data...")
    scaler        = joblib.load(os.path.join(CKPT, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(CKPT, "feature_names.pkl"))
    X_scaled      = pd.read_csv(os.path.join(PROC, x_file))
    y_raw         = pd.read_csv(os.path.join(PROC, y_file)).values.ravel()
    y             = y_raw - 1 if y_raw.min() > 0 else y_raw
    X_train_df    = pd.read_csv(os.path.join(PROC, "X_train.csv"))
    X_train       = X_train_df.values
    y_train_raw   = pd.read_csv(os.path.join(PROC, "y_train.csv")).values.ravel()
    y_train       = y_train_raw - 1 if y_train_raw.min() > 0 else y_train_raw
    good_mask     = np.isin(y_train, list(TARGET_CLASSES))
    good_samples  = X_train[good_mask]

    print(f"\n=== Evaluating 3-tier CounterfactualRCA ({split} split) ===")
    rca    = CounterfactualRCA()
    df_rca = evaluate("3-tier RCA", rca, X_scaled, y, scaler,
                      feature_names, X_train, good_samples)
    agg_rca = aggregate(df_rca)

    print(f"\n=== Evaluating centroid-ablation baseline ({split} split) ===")
    greedy    = GreedyCounterfactual()
    df_greedy = evaluate("Greedy", greedy, X_scaled, y, scaler,
                         feature_names, X_train, good_samples)
    agg_greedy = aggregate(df_greedy)

    mcn = run_mcnemar(df_rca, df_greedy)

    comparison = {
        "split":        split,
        "3-tier RCA":   {**agg_rca,    "mcnemar_p": mcn["mcnemar_p"]},
        "Greedy":       {**agg_greedy, "mcnemar_p": mcn["mcnemar_p"]},
        "mcnemar_test": mcn,
    }

    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2)

    df_all = pd.concat([df_rca, df_greedy], ignore_index=True)
    df_all.to_csv(out_csv, index=False)

    print("\n" + "=" * 68)
    print(f"  METHOD COMPARISON — {split.upper()} SPLIT")
    print("=" * 68)
    print(f"  {'Metric':<24} {'3-tier RCA':>16} {'Ablation':>16}")
    print(f"  {'-'*24} {'-'*16} {'-'*16}")
    rows_display = [
        ("Resolution rate",    f"{agg_rca['resolution_rate']:.1%}",
                               f"{agg_greedy['resolution_rate']:.1%}"),
        ("Validator rate",     f"{agg_rca['validator_rate']:.1%}",
                               f"{agg_greedy['validator_rate']:.1%}"),
        ("Mean proximity",     f"{agg_rca['mean_proximity']:.4f}",
                               f"{agg_greedy['mean_proximity']:.4f}"),
        ("Mean sparsity",      f"{agg_rca['mean_sparsity']:.1f} feat",
                               f"{agg_greedy['mean_sparsity']:.1f} feat"),
        ("Mean NN distance",   f"{agg_rca['mean_nn_distance']:.4f}",
                               f"{agg_greedy['mean_nn_distance']:.4f}"),
    ]
    for m, r, g in rows_display:
        print(f"  {m:<24} {r:>16} {g:>16}")
    print(f"\n  McNemar p-value: {mcn['mcnemar_p']:.4f}  "
          f"({mcn['interpretation']})")
    print(f"  Contingency: both={mcn['contingency']['a_both']}, "
          f"RCA-only={mcn['contingency']['c_rca_only']}, "
          f"ablation-only={mcn['contingency']['b_greedy_only']}, "
          f"neither={mcn['contingency']['d_neither']}")
    print(f"\n  Saved: {out_json}")
    print(f"  Saved: {out_csv}")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCA method comparison")
    parser.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Data split to evaluate (default: val)"
    )
    args = parser.parse_args()
    main(args.split)
