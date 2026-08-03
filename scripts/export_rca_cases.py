"""
export_rca_cases.py — Worked counterfactual cases and unresolved-case failure
analysis for thesis Chapter 4.

Case selection is driven entirely by models/rca_results_test.csv (already
produced by scripts/evaluate_rca.py --split test); this script re-runs
CounterfactualRCA.analyze() only on the three selected samples to recover the
full per-feature adjustment detail that the batch summary CSV does not store.
No new evaluation sweep is performed.

Outputs (thesis_assets/data/):
  counterfactual_cases.json      3 fully-worked cases (confirmed/unconfirmed/unresolved)
  counterfactual_case_table.csv  confirmed case, parameter/original/suggested/delta
  unresolved_analysis.json       7 unresolved vs. 140 resolved test-split cases

Usage:
    python scripts/export_rca_cases.py
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from scipy.stats import mannwhitneyu
from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

from counterfactual_rca import CounterfactualRCA, TARGET_CLASSES  # noqa: E402

CKPT = os.path.join(BASE, "models", "checkpoints")
PROC = os.path.join(BASE, "data", "processed")
OUT_DIR = os.path.join(BASE, "thesis_assets", "data")

CLASS_NAMES = {0: "Waste", 1: "Acceptable", 2: "Target", 3: "Inefficient"}
RESULTS_CSV = os.path.join(BASE, "models", "rca_results_test.csv")


def _load_split(name: str):
    X = pd.read_csv(os.path.join(PROC, f"X_{name}.csv"))
    y_raw = pd.read_csv(os.path.join(PROC, f"y_{name}.csv")).values.ravel()
    y = y_raw - 1 if y_raw.min() > 0 else y_raw
    return X, y


def _cf_full_vector(scaler, feature_names, x_real_row: np.ndarray, adjustments: list) -> np.ndarray:
    """Apply the RCA's suggested per-feature values onto the original real-unit
    vector (unmodified features keep their original value), then re-scale.
    Same reconstruction technique already used in scripts/compare_rca_methods.py."""
    cf_real = x_real_row.copy()
    for a in adjustments:
        fi = feature_names.index(a["feature"])
        cf_real[fi] = a["suggested"]
    return scaler.transform([cf_real])[0]


def _run_case(rca: CounterfactualRCA, scaler, feature_names,
              X_test_scaled: pd.DataFrame, idx: int) -> dict:
    x_scaled_row = X_test_scaled.iloc[[idx]]
    x_real_row = scaler.inverse_transform(x_scaled_row)[0]
    x_real_df = pd.DataFrame([x_real_row], columns=feature_names)

    result = rca.analyze(x_real_df)

    entry = {
        "sample_index_in_test_split": idx,
        "tier": result["tier"],
        "status": result["status"],
        "message": result["message"],
        "validator_ok": bool(result["validator_ok"]),
        "original": {
            "feature_values": {f: float(v) for f, v in zip(feature_names, x_real_row)},
            "probability_vector": {CLASS_NAMES[i]: float(p) for i, p in enumerate(result["proba"])},
            "confidence_acc_plus_target": round(result["confidence"], 4),
        },
    }

    if result["status"] == "resolved":
        cf_scaled = _cf_full_vector(scaler, feature_names, x_real_row, result["adjustments"])
        cf_proba = rca.model.predict_proba(cf_scaled.reshape(1, -1))[0]
        entry["counterfactual"] = {
            "tier_method": "Tier 1 (SHAP+LIME)" if result["tier"] == 1
                           else "Tier 2 (NN-anchored)",
            "adjustments": result["adjustments"],
            "probability_vector": {CLASS_NAMES[i]: float(p) for i, p in enumerate(cf_proba)},
            "confidence_acc_plus_target": round(result.get("cf_confidence", 0.0), 4),
        }

    return entry


def select_case(df: pd.DataFrame, tier_ok, validator_ok, rca, scaler, feature_names,
                 X_test_scaled, label: str) -> dict:
    candidates = df[df["tier"].apply(tier_ok)]
    if validator_ok is not None:
        candidates = candidates[candidates["validator_ok"] == validator_ok]
    if candidates.empty:
        raise RuntimeError(f"No candidate sample found in rca_results_test.csv for case '{label}'.")

    for _, row in candidates.iterrows():
        idx = int(row["sample_idx"])
        entry = _run_case(rca, scaler, feature_names, X_test_scaled, idx)
        matches_tier = tier_ok(entry["tier"])
        matches_validator = (validator_ok is None) or (entry["validator_ok"] == validator_ok)
        if matches_tier and matches_validator:
            entry["true_class"] = int(row["true_class"])
            entry["true_label"] = row["true_label"]
            print(f"  {label}: sample_idx={idx}  tier={entry['tier']}  "
                  f"validator_ok={entry['validator_ok']}")
            return entry
        print(f"  {label}: candidate sample_idx={idx} did not reproduce the expected "
              f"outcome on re-run (tier={entry['tier']}, validator_ok={entry['validator_ok']}); "
              "trying next candidate.")

    raise RuntimeError(f"No candidate for case '{label}' reproduced the expected outcome on re-run.")


def export_counterfactual_cases(rca, scaler, feature_names, X_test_scaled, df) -> dict:
    confirmed = select_case(df, lambda t: t in (1, 2), True, rca, scaler, feature_names,
                             X_test_scaled, "validator-confirmed")
    unconfirmed = select_case(df, lambda t: t in (1, 2), False, rca, scaler, feature_names,
                               X_test_scaled, "validator-unconfirmed")
    # "Unresolved" = excluded from the resolved count in evaluate_rca.py's own
    # accounting (resolved = tier1 + tier2). On the test split this bucket
    # turns out to be entirely tier-0 ("already_acceptable") rows, not tier-3
    # escalations: there are zero tier-3 cases in the test split. These are
    # samples whose TRUE label is non-conforming but whose model confidence
    # for {Acceptable, Target} already exceeds the 0.55 acceptance threshold,
    # so the RCA engine never attempts a correction. See unresolved_analysis.json.
    unresolved = select_case(df, lambda t: t not in (1, 2), None, rca, scaler, feature_names,
                              X_test_scaled, "unresolved")

    result = {
        "split": "test",
        "cases": {
            "validator_confirmed": confirmed,
            "validator_unconfirmed": unconfirmed,
            "unresolved": unresolved,
        },
    }
    with open(os.path.join(OUT_DIR, "counterfactual_cases.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def export_case_table(cases: dict) -> None:
    confirmed = cases["cases"]["validator_confirmed"]
    rows = []
    for a in confirmed["counterfactual"]["adjustments"]:
        rows.append({
            "Parameter": a["feature"],
            "Original": a["current"],
            "Suggested": a["suggested"],
            "Delta": a["delta"],
            "Direction": a["direction"],
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "counterfactual_case_table.csv"), index=False
    )


def export_unresolved_analysis(df, scaler, feature_names, X_test_scaled, X_train_scaled, y_train) -> dict:
    model = joblib.load(os.path.join(CKPT, "current_model.pkl"))
    good_mask = np.isin(y_train, list(TARGET_CLASSES))
    good_samples = X_train_scaled.values[good_mask]

    resolved_idx = df[df["tier"].isin([1, 2])]["sample_idx"].astype(int).tolist()
    unresolved_mask = ~df["tier"].isin([1, 2])
    unresolved_idx = df[unresolved_mask]["sample_idx"].astype(int).tolist()
    unresolved_tier_counts = df[unresolved_mask]["tier"].value_counts().to_dict()
    unresolved_status_counts = df[unresolved_mask]["status"].value_counts().to_dict()

    def _group_stats(idx_list):
        rows_real = []
        confidences = []
        nn_dists = []
        for idx in idx_list:
            x_scaled = X_test_scaled.iloc[idx].values
            x_real = scaler.inverse_transform([x_scaled])[0]
            rows_real.append(x_real)
            proba = model.predict_proba(x_scaled.reshape(1, -1))[0]
            confidences.append(float(proba[1] + proba[2]))
            nn_dists.append(float(euclidean_distances([x_scaled], good_samples).min()))
        real_df = pd.DataFrame(rows_real, columns=feature_names)
        return real_df, np.array(confidences), np.array(nn_dists)

    resolved_real, resolved_conf, resolved_nn = _group_stats(resolved_idx)
    unresolved_real, unresolved_conf, unresolved_nn = _group_stats(unresolved_idx)

    feature_comparison = {}
    for f in feature_names:
        feature_comparison[f] = {
            "resolved_mean": round(float(resolved_real[f].mean()), 4),
            "resolved_std": round(float(resolved_real[f].std(ddof=1)), 4),
            "unresolved_mean": round(float(unresolved_real[f].mean()), 4),
            "unresolved_std": round(float(unresolved_real[f].std(ddof=1)), 4),
        }

    conf_u, conf_p = mannwhitneyu(unresolved_conf, resolved_conf, alternative="two-sided")
    nn_u, nn_p = mannwhitneyu(unresolved_nn, resolved_nn, alternative="two-sided")

    result = {
        "split": "test",
        "n_resolved": len(resolved_idx),
        "n_unresolved": len(unresolved_idx),
        "unresolved_tier_counts": {int(k): int(v) for k, v in unresolved_tier_counts.items()},
        "unresolved_status_counts": {str(k): int(v) for k, v in unresolved_status_counts.items()},
        "note": (
            "On the test split, all 7 'unresolved' cases (excluded from the "
            "resolved=tier1+tier2 count) are tier-0 'already_acceptable' rows, "
            "not tier-3 escalations -- there are zero tier-3 cases in this "
            "split. Each of the 7 has a TRUE label of Waste or Inefficient but "
            "a model confidence P(Acceptable)+P(Target) already >= 0.55, so "
            "the RCA engine never attempts a correction. This reflects "
            "disagreement between ground-truth label and model confidence on "
            "these specific cycles, not a failure of the counterfactual search."
        ),
        "confidence_acc_plus_target": {
            "resolved_mean": round(float(resolved_conf.mean()), 4),
            "resolved_std": round(float(resolved_conf.std(ddof=1)), 4),
            "unresolved_mean": round(float(unresolved_conf.mean()), 4),
            "unresolved_std": round(float(unresolved_conf.std(ddof=1)), 4),
            "mannwhitney_u": round(float(conf_u), 4),
            "mannwhitney_p": round(float(conf_p), 6),
        },
        "nn_distance_to_conforming_manifold": {
            "definition": "L2 distance (scaled feature space) from the original "
                           "defective sample to its nearest conforming (Acceptable "
                           "or Target) training sample",
            "resolved_mean": round(float(resolved_nn.mean()), 4),
            "resolved_std": round(float(resolved_nn.std(ddof=1)), 4),
            "unresolved_mean": round(float(unresolved_nn.mean()), 4),
            "unresolved_std": round(float(unresolved_nn.std(ddof=1)), 4),
            "mannwhitney_u": round(float(nn_u), 4),
            "mannwhitney_p": round(float(nn_p), 6),
        },
        "feature_comparison": feature_comparison,
        "unresolved_sample_indices": unresolved_idx,
    }
    with open(os.path.join(OUT_DIR, "unresolved_analysis.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Worked counterfactual cases + unresolved-case analysis ===")

    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Required source artifact missing: {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)

    scaler = joblib.load(os.path.join(CKPT, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(CKPT, "feature_names.pkl"))
    X_test_scaled, _ = _load_split("test")
    X_train_scaled, y_train = _load_split("train")

    print("Initialising CounterfactualRCA (default params, matches evaluate_rca.py)...")
    rca = CounterfactualRCA()

    print("\nSelecting worked cases...")
    cases = export_counterfactual_cases(rca, scaler, feature_names, X_test_scaled, df)

    print("\nExporting counterfactual_case_table.csv (confirmed case)...")
    export_case_table(cases)

    print("\nAnalysing 7 unresolved vs. 140 resolved test-split cases...")
    unresolved_result = export_unresolved_analysis(
        df, scaler, feature_names, X_test_scaled, X_train_scaled, y_train
    )
    print(f"  confidence: resolved={unresolved_result['confidence_acc_plus_target']['resolved_mean']}  "
          f"unresolved={unresolved_result['confidence_acc_plus_target']['unresolved_mean']}  "
          f"p={unresolved_result['confidence_acc_plus_target']['mannwhitney_p']}")
    print(f"  nn_distance: resolved={unresolved_result['nn_distance_to_conforming_manifold']['resolved_mean']}  "
          f"unresolved={unresolved_result['nn_distance_to_conforming_manifold']['unresolved_mean']}  "
          f"p={unresolved_result['nn_distance_to_conforming_manifold']['mannwhitney_p']}")

    print(f"\nDone. Files written to {OUT_DIR}")


if __name__ == "__main__":
    main()
