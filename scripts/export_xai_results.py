"""
export_xai_results.py — XAI batch export for thesis Chapter 4.

Every number here is produced by calling the existing XAIEngine (SHAP
TreeExplainer + LIME TabularExplainer) against real data already on disk.
No SHAP or LIME logic is reimplemented; this script only orchestrates calls
to xai_engine.XAIEngine and aggregates/serialises the results.

Outputs (thesis_assets/data/):
  shap_global_importance.json   mean |SHAP| per feature, aggregated + per class
  shap_vs_impurity.json         SHAP rank vs RF feature_importances_, Spearman
  shap_vs_filters.json          SHAP rank vs Relief/ANOVA (Polenta et al. 2022)
  local_explanations.json       3 worked test-split cases (Waste/Inefficient/Target)
  shap_lime_agreement.json      SHAP-LIME rank agreement over 50 test samples
  lime_stability.json           LIME repeatability, 10 runs on one fixed sample
  xai_timing.json               wall-clock timing + LIME cache hit rate

Usage:
    python scripts/export_xai_results.py

Run time: ~5-15 minutes (dominated by the 50-sample LIME agreement study and
the SHAP pass over the full training split).
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

from xai_engine import XAIEngine, CLASS_NAMES  # noqa: E402

CKPT = os.path.join(BASE, "models", "checkpoints")
PROC = os.path.join(BASE, "data", "processed")
OUT_DIR = os.path.join(BASE, "thesis_assets", "data")

N_LIME_AGREEMENT_SAMPLES = 50
N_LIME_STABILITY_RUNS = 10
N_TIMING_SAMPLES = 15  # unique samples timed for SHAP/LIME wall-clock

# Relief and ANOVA normalised feature weights, Polenta et al. (2022),
# Figures 2 and 3. Reproduced verbatim as reference constants; not derived
# from this dataset.
POLENTA_RELIEF = {
    "cycle time": 1.0,
    "filling time": 0.5749,
    "clamping force": 0.1849,
    "closing force": 0.1793,
    "screw position": 0.1421,
    "shot volume": 0.1393,
    "mold temperature": 0.1325,
    "plasticizing time": 0.0913,
    "specific injection pressure": 0.0632,
    "torque peak": 0.0611,
    "torque mean": 0.0584,
    "specific back pressure": 0.0038,
    "melt temperature": 0.0,
}
POLENTA_ANOVA = {
    "cycle time": 1.0,
    "specific injection pressure": 0.0295,
    "mold temperature": 0.0289,
    "filling time": 0.025,
    "shot volume": 0.0204,
    "screw position": 0.0202,
    "plasticizing time": 0.0199,
    "closing force": 0.0145,
    "clamping force": 0.0101,
    "specific back pressure": 0.0032,
    "torque mean": 0.002,
    "torque peak": 0.0012,
    "melt temperature": 0.0,
}
# Explicit mapping: Polenta et al. (2022) parameter name -> this dataset's column.
PAPER_TO_COLUMN = {
    "cycle time": "ZUx - Cycle time",
    "filling time": "time_to_fill",
    "clamping force": "SKs - Clamping force peak value",
    "closing force": "SKx - Closing force",
    "screw position": "CPn - Screw position at the end of hold pressure",
    "shot volume": "SVo - Shot volume",
    "mold temperature": "Mold temperature",
    "plasticizing time": "ZDx - Plasticizing time",
    "specific injection pressure": "APVs - Specific injection pressure peak value",
    "torque peak": "Ms - Torque peak value current cycle",
    "torque mean": "Mm - Torque mean value current cycle",
    "specific back pressure": "APSs - Specific back pressure peak value",
    "melt temperature": "Melt temperature",
}


def _load_split(name: str):
    X = pd.read_csv(os.path.join(PROC, f"X_{name}.csv"))
    y_raw = pd.read_csv(os.path.join(PROC, f"y_{name}.csv")).values.ravel()
    y = y_raw - 1 if y_raw.min() > 0 else y_raw
    return X, y


def _shap_full(engine: XAIEngine, X: pd.DataFrame) -> np.ndarray:
    """(n_samples, n_features, n_classes) SHAP array via engine's own explainer."""
    sv = np.array(engine.explainer.shap_values(X))
    if sv.ndim != 3:
        raise RuntimeError(f"Unexpected SHAP output shape {sv.shape}; "
                            "expected (n_samples, n_features, n_classes).")
    return sv


def _rank_desc(values: dict) -> list:
    return [k for k, _ in sorted(values.items(), key=lambda kv: kv[1], reverse=True)]


def export_shap_global_importance(engine: XAIEngine, X_train: pd.DataFrame) -> dict:
    sv = _shap_full(engine, X_train)  # (n, f, c)
    feats = engine.feature_names

    agg = {feats[i]: float(np.mean(np.abs(sv[:, i, :]))) for i in range(len(feats))}
    per_class = {}
    for c_idx, cls_name in enumerate(CLASS_NAMES):
        per_class[cls_name] = {
            feats[i]: float(np.mean(np.abs(sv[:, i, c_idx]))) for i in range(len(feats))
        }

    result = {
        "split": "train",
        "n_samples": int(len(X_train)),
        "aggregated_across_classes": agg,
        "rank_aggregated": _rank_desc(agg),
        "per_class": {
            cls: {"importance": vals, "rank": _rank_desc(vals)}
            for cls, vals in per_class.items()
        },
    }
    with open(os.path.join(OUT_DIR, "shap_global_importance.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result, sv


def export_shap_vs_impurity(engine: XAIEngine, shap_global: dict) -> dict:
    feats = engine.feature_names
    shap_imp = shap_global["aggregated_across_classes"]
    impurity_imp = {f: float(v) for f, v in zip(feats, engine.model.feature_importances_)}

    shap_vec = [shap_imp[f] for f in feats]
    impurity_vec = [impurity_imp[f] for f in feats]
    rho, p = spearmanr(shap_vec, impurity_vec)

    result = {
        "shap_importance": shap_imp,
        "impurity_importance": impurity_imp,
        "shap_rank": _rank_desc(shap_imp),
        "impurity_rank": _rank_desc(impurity_imp),
        "spearman_r": round(float(rho), 4),
        "spearman_p": round(float(p), 6),
    }
    with open(os.path.join(OUT_DIR, "shap_vs_impurity.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def export_shap_vs_filters(shap_global: dict) -> dict:
    shap_imp = shap_global["aggregated_across_classes"]

    # Build aligned vectors in a fixed paper-parameter order.
    paper_params = list(PAPER_TO_COLUMN.keys())
    shap_vec = [shap_imp[PAPER_TO_COLUMN[p]] for p in paper_params]
    relief_vec = [POLENTA_RELIEF[p] for p in paper_params]
    anova_vec = [POLENTA_ANOVA[p] for p in paper_params]

    rho_relief, p_relief = spearmanr(shap_vec, relief_vec)
    rho_anova, p_anova = spearmanr(shap_vec, anova_vec)

    shap_top1 = paper_params[int(np.argmax(shap_vec))]
    relief_top1 = paper_params[int(np.argmax(relief_vec))]
    anova_top1 = paper_params[int(np.argmax(anova_vec))]

    result = {
        "parameter_mapping": PAPER_TO_COLUMN,
        "shap_importance_by_paper_name": dict(zip(paper_params, shap_vec)),
        "relief_weight": POLENTA_RELIEF,
        "anova_weight": POLENTA_ANOVA,
        "spearman_shap_vs_relief": {"r": round(float(rho_relief), 4),
                                     "p": round(float(p_relief), 6)},
        "spearman_shap_vs_anova": {"r": round(float(rho_anova), 4),
                                    "p": round(float(p_anova), 6)},
        "top1_feature": {"shap": shap_top1, "relief": relief_top1, "anova": anova_top1},
        "agreement_note": (
            f"SHAP top-ranked parameter is '{shap_top1}'; Relief top-ranked is "
            f"'{relief_top1}'; ANOVA top-ranked is '{anova_top1}'. "
            f"Spearman r(SHAP, Relief)={rho_relief:.3f}, "
            f"r(SHAP, ANOVA)={rho_anova:.3f}."
        ),
        "source": "Polenta et al. (2022), Figures 2-3 (Relief/ANOVA weights, "
                  "reproduced as reference constants)",
    }
    with open(os.path.join(OUT_DIR, "shap_vs_filters.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def _pick_case(X_real: pd.DataFrame, y: np.ndarray, model, target_class: int, require_correct: bool):
    preds = model.predict(joblib.load(os.path.join(CKPT, "scaler.pkl")).transform(X_real))
    for idx in range(len(y)):
        if y[idx] != target_class:
            continue
        if require_correct and preds[idx] != target_class:
            continue
        return idx, int(preds[idx])
    raise RuntimeError(f"No sample found for class {target_class} "
                        f"(require_correct={require_correct}) in the test split.")


def export_local_explanations(engine: XAIEngine, X_test_scaled: pd.DataFrame,
                               X_test_real: pd.DataFrame, y_test: np.ndarray) -> dict:
    scaler = joblib.load(os.path.join(CKPT, "scaler.pkl"))
    cases_spec = [
        ("waste_case", 0, True),
        ("inefficient_case", 3, True),
        ("target_case", 2, True),
    ]

    cases = {}
    for name, cls, require_correct in cases_spec:
        idx, pred = _pick_case(X_test_real, y_test, engine.model, cls, require_correct)
        row_scaled = X_test_scaled.iloc[[idx]]
        row_real = X_test_real.iloc[[idx]]

        proba = engine.model.predict_proba(row_scaled.values)[0]
        sv = _shap_full(engine, row_scaled)[0, :, pred]  # SHAP for predicted class
        lime_df = engine.get_full_lime_explanation(row_scaled, label=pred)

        cases[name] = {
            "sample_index_in_test_split": int(idx),
            "true_class": int(y_test[idx]),
            "true_label": CLASS_NAMES[int(y_test[idx])],
            "predicted_class": pred,
            "predicted_label": CLASS_NAMES[pred],
            "probability_vector": {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)},
            "raw_feature_values": {
                f: float(row_real.iloc[0][f]) for f in engine.feature_names
            },
            "shap_contribution": {
                f: float(sv[i]) for i, f in enumerate(engine.feature_names)
            },
            "lime_coefficient": {
                r["Feature"]: float(r["LIME_Coefficient"])
                for r in lime_df.to_dict("records")
            },
        }

    result = {"split": "test", "cases": cases}
    with open(os.path.join(OUT_DIR, "local_explanations.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def export_shap_lime_agreement(engine: XAIEngine, X_test_scaled: pd.DataFrame,
                                model) -> dict:
    n = min(N_LIME_AGREEMENT_SAMPLES, len(X_test_scaled))
    rng = np.random.RandomState(0)
    idxs = rng.choice(len(X_test_scaled), size=n, replace=False)

    rhos, overlaps = [], []
    per_sample = []
    for idx in idxs:
        row = X_test_scaled.iloc[[idx]]
        pred = int(model.predict(row.values)[0])

        sv = _shap_full(engine, row)[0, :, pred]
        shap_rank = [engine.feature_names[i] for i in np.argsort(-np.abs(sv))]

        lime_df = engine.get_full_lime_explanation(row, label=pred)
        lime_rank = lime_df["Feature"].tolist()

        # Align LIME rank onto the SHAP feature order for Spearman comparison.
        shap_order = {f: r for r, f in enumerate(shap_rank)}
        lime_order = {f: r for r, f in enumerate(lime_rank)}
        common = engine.feature_names
        rho, _ = spearmanr([shap_order[f] for f in common], [lime_order[f] for f in common])

        top5_overlap = len(set(shap_rank[:5]) & set(lime_rank[:5]))

        rhos.append(float(rho))
        overlaps.append(top5_overlap)
        per_sample.append({"sample_index": int(idx), "predicted_class": pred,
                            "spearman_r": round(float(rho), 4),
                            "top5_overlap": top5_overlap})

    result = {
        "split": "test",
        "n_samples": n,
        "spearman_r_mean": round(float(np.mean(rhos)), 4),
        "spearman_r_std": round(float(np.std(rhos, ddof=1)), 4),
        "top5_overlap_mean": round(float(np.mean(overlaps)), 4),
        "top5_overlap_std": round(float(np.std(overlaps, ddof=1)), 4),
        "per_sample": per_sample,
    }
    with open(os.path.join(OUT_DIR, "shap_lime_agreement.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def _parse_lime_explanation(exp, label: int, feature_names: list) -> dict:
    """Map a raw lime explanation (as_list) onto this dataset's feature names.

    Mirrors the small matching rule already used internally by
    XAIEngine._lime_raw / get_full_lime_explanation — consuming LIME's output
    format, not reimplementing the LIME algorithm.
    """
    coeff_map = {}
    for fname_expr, coeff in exp.as_list(label=label):
        for fname in feature_names:
            if fname == fname_expr or fname in fname_expr:
                coeff_map[fname] = float(coeff)
                break
    return {f: coeff_map.get(f, 0.0) for f in feature_names}


def export_lime_stability(engine: XAIEngine, X_test_scaled: pd.DataFrame,
                           model) -> dict:
    idx = 0  # first test-split sample: fixed, arbitrary but reproducible
    row = X_test_scaled.iloc[[idx]]
    pred = int(model.predict(row.values)[0])

    runs = []
    for _ in range(N_LIME_STABILITY_RUNS):
        # Bypass XAIEngine's result cache deliberately: the cache is a UI
        # optimisation keyed on (sample, label, num_samples), so repeated
        # calls with identical arguments would return the SAME cached result
        # rather than re-running LIME's stochastic perturbation sampling.
        # We call the engine's own lime_explainer instance directly instead.
        exp = engine.lime_explainer.explain_instance(
            row.values[0], engine.model.predict_proba,
            num_features=len(engine.feature_names), labels=[pred], num_samples=2000,
        )
        runs.append(_parse_lime_explanation(exp, pred, engine.feature_names))

    df = pd.DataFrame(runs)  # columns = features, rows = runs
    coeff_std = {f: float(df[f].std(ddof=1)) for f in engine.feature_names}

    top5_sets = [
        set(df.iloc[r].abs().sort_values(ascending=False).index[:5])
        for r in range(len(df))
    ]
    from collections import Counter
    top5_counter = Counter(f for s in top5_sets for f in s)
    reference_top5 = set(top5_sets[0])
    stability_vs_first_run = [
        len(reference_top5 & s) for s in top5_sets
    ]

    result = {
        "sample_index_in_test_split": idx,
        "predicted_class": pred,
        "n_runs": N_LIME_STABILITY_RUNS,
        "coefficient_std_by_feature": coeff_std,
        "top5_feature_frequency_across_runs": dict(top5_counter),
        "top5_overlap_with_first_run": stability_vs_first_run,
        "top5_overlap_with_first_run_mean": round(float(np.mean(stability_vs_first_run)), 4),
    }
    with open(os.path.join(OUT_DIR, "lime_stability.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def export_xai_timing(model) -> dict:
    """Uses a freshly-constructed XAIEngine so the LIME cache starts empty —
    otherwise entries left behind by the earlier export steps (which share an
    engine instance) would inflate the hit rate beyond what this isolated
    session actually produces."""
    n = min(N_TIMING_SAMPLES, 291)
    fresh_engine = XAIEngine()
    X_test_scaled, _ = _load_split("test")
    idxs = list(range(n))

    shap_times = []
    for idx in idxs:
        row = X_test_scaled.iloc[[idx]]
        t0 = time.perf_counter()
        _shap_full(fresh_engine, row)
        shap_times.append(time.perf_counter() - t0)

    lime_times = []
    calls_made = 0
    cache_hits = 0
    for idx in idxs:
        row = X_test_scaled.iloc[[idx]]
        pred = int(model.predict(row.values)[0])
        for repeat in range(2):  # each sample queried twice: 1st miss, 2nd hit
            t0 = time.perf_counter()
            fresh_engine.get_lime_directions(row, fresh_engine.feature_names, label=pred)
            lime_times.append(time.perf_counter() - t0)
            calls_made += 1
            if repeat == 1:
                cache_hits += 1
    cache_misses = calls_made - cache_hits
    cache_hit_rate = cache_hits / calls_made if calls_made else 0.0

    result = {
        "n_shap_calls": n,
        "shap_call_seconds_mean": round(float(np.mean(shap_times)), 5),
        "shap_call_seconds_std": round(float(np.std(shap_times, ddof=1)), 5),
        "n_lime_calls": calls_made,
        "lime_call_seconds_mean": round(float(np.mean(lime_times)), 5),
        "lime_call_seconds_std": round(float(np.std(lime_times, ddof=1)), 5),
        "session_definition": (
            f"{n} unique defect samples, each queried twice in sequence "
            "(simulating a user re-opening a previously viewed cycle)"
        ),
        "cache_hit_rate": round(float(cache_hit_rate), 4),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
    }
    with open(os.path.join(OUT_DIR, "xai_timing.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== XAI batch export ===")
    t_start = time.time()

    engine = XAIEngine()
    scaler = joblib.load(os.path.join(CKPT, "scaler.pkl"))

    X_train, _ = _load_split("train")
    X_test, y_test = _load_split("test")
    X_test_real = pd.DataFrame(scaler.inverse_transform(X_test), columns=engine.feature_names)

    print("\n[1/7] SHAP global importance (full training split)...")
    shap_global, _ = export_shap_global_importance(engine, X_train)
    print(f"  top-3 (aggregated): {shap_global['rank_aggregated'][:3]}")

    print("\n[2/7] SHAP vs. RF impurity importance...")
    imp_result = export_shap_vs_impurity(engine, shap_global)
    print(f"  Spearman r = {imp_result['spearman_r']}")

    print("\n[3/7] SHAP vs. Relief/ANOVA (Polenta et al. 2022)...")
    filt_result = export_shap_vs_filters(shap_global)
    print(f"  {filt_result['agreement_note']}")

    print("\n[4/7] Local explanations (3 worked test-split cases)...")
    export_local_explanations(engine, X_test, X_test_real, y_test)

    print(f"\n[5/7] SHAP-LIME agreement ({N_LIME_AGREEMENT_SAMPLES} test samples)...")
    agree_result = export_shap_lime_agreement(engine, X_test, engine.model)
    print(f"  mean Spearman r = {agree_result['spearman_r_mean']} "
          f"(std {agree_result['spearman_r_std']})")

    print(f"\n[6/7] LIME stability ({N_LIME_STABILITY_RUNS} runs, fixed sample)...")
    export_lime_stability(engine, X_test, engine.model)

    print(f"\n[7/7] XAI timing ({N_TIMING_SAMPLES} samples, isolated engine)...")
    timing_result = export_xai_timing(engine.model)
    print(f"  SHAP {timing_result['shap_call_seconds_mean']}s/call, "
          f"LIME {timing_result['lime_call_seconds_mean']}s/call, "
          f"cache hit rate {timing_result['cache_hit_rate']:.1%}")

    print(f"\nDone in {time.time() - t_start:.0f}s. Files written to {OUT_DIR}")


if __name__ == "__main__":
    main()
