"""
export_thesis_tables.py — Table export for thesis Chapters 3-4.

Every table reads its numbers from an artifact already on disk (this
repository's models/*.json/csv, or thesis_assets/data/*.json produced by the
export_*.py scripts run earlier in the pipeline). No result value is
hardcoded. Where a required source input does not exist (the Polenta et al.
2022 Table 10 per-class baseline), the table is skipped with a loud printed
explanation rather than fabricated -- see tbl_ch4_baseline_comparison.

Each table is written to thesis_assets/tables/csv/<name>.csv AND
thesis_assets/tables/markdown/<name>.md.

Usage:
    python scripts/export_thesis_tables.py
"""

import os
import re
import json

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "thesis_assets", "data")
MODELS_DIR = os.path.join(BASE, "models")
DOCS_DIR = os.path.join(BASE, "docs")
CSV_DIR = os.path.join(BASE, "thesis_assets", "tables", "csv")
MD_DIR = os.path.join(BASE, "thesis_assets", "tables", "markdown")

CLASS_NAMES = ["Waste", "Acceptable", "Target", "Inefficient"]

# Polenta et al. (2022), Table 10, Random Forest row -- reproduced verbatim
# as a reference constant (percentages converted to fractions). See the
# identical constant and its provenance note in generate_thesis_figures.py.
POLENTA_TABLE10_RF = {
    "Waste":       {"precision": 0.9722, "recall": 0.9459},
    "Acceptable":  {"precision": 0.9511, "recall": 0.9581},
    "Target":      {"precision": 0.9437, "recall": 0.9194},
    "Inefficient": {"precision": 0.9342, "recall": 0.9736},
}


def _load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required source artifact missing: {path}")
    with open(path) as f:
        return json.load(f)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured-markdown table writer (avoids adding the
    `tabulate` dependency for this alone)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(os.path.join(CSV_DIR, f"{name}.csv"), index=False)
    with open(os.path.join(MD_DIR, f"{name}.md"), "w") as f:
        f.write(_df_to_markdown(df))
    print(f"  saved {name}.csv / .md ({len(df)} rows)")


# ── Chapter 3 ────────────────────────────────────────────────────────────────

def tbl_ch3_feature_specification():
    d = _load_json(os.path.join(DATA_DIR, "feature_specification.json"))
    rows = []
    for feat, spec in d["features"].items():
        rows.append({
            "Parameter": feat,
            "Physical quantity": spec["physical_quantity"],
            "Unit": spec["unit"],
            "Min": spec["min"], "Max": spec["max"],
            "Mean": spec["mean"], "Median": spec["median"], "Std": spec["std"],
        })
    save_table(pd.DataFrame(rows), "tbl_ch3_feature_specification")


def tbl_ch3_class_distribution():
    d = _load_json(os.path.join(DATA_DIR, "class_distribution.json"))
    rows = []
    for scope in ["full_dataset"] + list(d["splits"].keys()):
        dist = d["full_dataset"] if scope == "full_dataset" else d["splits"][scope]
        row = {"Scope": scope, "N": dist["n_total"]}
        for c in CLASS_NAMES:
            row[f"{c} (n)"] = dist["by_class"][c]["count"]
            row[f"{c} (%)"] = dist["by_class"][c]["pct"]
        row["Conforming (%)"] = dist["conforming"]["pct"]
        rows.append(row)
    save_table(pd.DataFrame(rows), "tbl_ch3_class_distribution")


def tbl_ch3_split_summary():
    d = _load_json(os.path.join(DATA_DIR, "split_summary.json"))
    rows = [{
        "Split": name, "N": d["sizes"][name], "Proportion": d["proportions"][name],
        "Random state": d["random_state"], "Stratified": d["stratified"],
        "Stratification confirmed": d["stratification_confirmed"],
    } for name in d["sizes"]]
    save_table(pd.DataFrame(rows), "tbl_ch3_split_summary")


def tbl_ch3_hyperparameter_provenance():
    path = os.path.join(DOCS_DIR, "hyperparameter_provenance.md")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required source artifact missing: {path}")
    with open(path) as f:
        text = f.read()

    match = re.search(
        r"\| Hyperparameter \|.*?\n((?:\|.*\n)+)", text
    )
    if not match:
        raise RuntimeError(
            f"Could not locate the hyperparameter mapping table in {path}; "
            "its format may have changed."
        )
    table_block = match.group(0)
    lines = [ln for ln in table_block.strip().split("\n") if ln.strip().startswith("|")]
    header = [c.strip().strip("*") for c in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:  # skip header + separator
        cells = [c.strip().replace("**", "") for c in ln.strip("|").split("|")]
        rows.append(dict(zip(header, cells)))
    save_table(pd.DataFrame(rows), "tbl_ch3_hyperparameter_provenance")


def tbl_ch3_tree_depth():
    d = _load_json(os.path.join(MODELS_DIR, "tree_depth.json"))
    row = {
        "n_estimators": d["n_estimators"],
        "Configured max_depth": d["max_depth_configured"],
        "Realised max depth": d["max_depth_realised"],
        "Realised mean depth": d["mean_depth_realised"],
        "Realised median depth": d["median_depth_realised"],
        "Realised min depth": d["min_depth_realised"],
        "Constraint binding?": d["binding"],
    }
    save_table(pd.DataFrame([row]), "tbl_ch3_tree_depth")


# ── Chapter 4 ────────────────────────────────────────────────────────────────

def tbl_ch4_algorithm_comparison():
    with open(os.path.join(MODELS_DIR, "experiments.json")) as f:
        runs = json.load(f)
    df = pd.DataFrame(runs)[["algo", "cv_mean", "cv_std", "val_acc", "val_f1"]]
    df = df.sort_values("cv_mean", ascending=False).rename(columns={
        "algo": "Algorithm", "cv_mean": "CV mean", "cv_std": "CV std",
        "val_acc": "Val accuracy", "val_f1": "Val F1 (weighted)",
    })
    save_table(df, "tbl_ch4_algorithm_comparison")


def tbl_ch4_test_performance():
    d = _load_json(os.path.join(MODELS_DIR, "test_evaluation.json"))
    row = {
        "Model": d["model"], "N samples": d["n_samples"],
        "Accuracy": d["accuracy"], "F1 (weighted)": d["f1_weighted"],
        "F1 (macro)": d["f1_macro"],
        "Non-conformance precision": d["non_conformance_precision"],
        "Non-conformance recall": d["non_conformance_recall"],
    }
    save_table(pd.DataFrame([row]), "tbl_ch4_test_performance")


def tbl_ch4_per_class_metrics():
    d = _load_json(os.path.join(MODELS_DIR, "test_evaluation.json"))
    rows = [{"Class": c, "Precision": v["precision"], "Recall": v["recall"],
             "Support": v["n_samples"]} for c, v in d["per_class"].items()]
    save_table(pd.DataFrame(rows), "tbl_ch4_per_class_metrics")


def tbl_ch4_confusion_matrix():
    d = _load_json(os.path.join(MODELS_DIR, "test_evaluation.json"))
    cm = np.array(d["confusion_matrix"])
    df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    df["Row total"] = df.sum(axis=1)
    col_totals = df.sum(axis=0)
    col_totals.name = "Column total"
    df = pd.concat([df, col_totals.to_frame().T])
    df = df.reset_index().rename(columns={"index": "True \\ Predicted"})
    save_table(df, "tbl_ch4_confusion_matrix")


def tbl_ch4_baseline_comparison():
    d = _load_json(os.path.join(MODELS_DIR, "test_evaluation.json"))
    this_work = d["per_class"]

    rows = []
    for c in CLASS_NAMES:
        rows.append({
            "Class": c,
            "This work: precision (test, n=291)": round(this_work[c]["precision"], 4),
            "Polenta (2022) RF: precision (5-fold CV, n=1451)": POLENTA_TABLE10_RF[c]["precision"],
            "This work: recall (test, n=291)": round(this_work[c]["recall"], 4),
            "Polenta (2022) RF: recall (5-fold CV, n=1451)": POLENTA_TABLE10_RF[c]["recall"],
        })
    df = pd.DataFrame(rows)
    df["_note"] = ""
    df.loc[0, "_note"] = (
        "Protocol difference: this work reports a single held-out test split "
        "(n=291); Polenta et al. report samples summed across the 5 folds of "
        "stratified 5-fold CV over the full 1451-sample dataset. Not a like-"
        "for-like comparison -- see docs/hyperparameter_provenance.md Sec. 5."
    )
    df = df.rename(columns={"_note": "Note"})
    save_table(df, "tbl_ch4_baseline_comparison")


def tbl_ch4_shap_global_ranking():
    d = _load_json(os.path.join(DATA_DIR, "shap_global_importance.json"))
    imp = d["aggregated_across_classes"]
    rows = [{"Rank": i + 1, "Feature": f, "Mean |SHAP|": imp[f]}
            for i, f in enumerate(d["rank_aggregated"])]
    save_table(pd.DataFrame(rows), "tbl_ch4_shap_global_ranking")


def tbl_ch4_importance_method_comparison():
    imp_data = _load_json(os.path.join(DATA_DIR, "shap_vs_impurity.json"))
    filt_data = _load_json(os.path.join(DATA_DIR, "shap_vs_filters.json"))
    mapping = filt_data["parameter_mapping"]  # paper name -> column
    inv_mapping = {v: k for k, v in mapping.items()}

    shap_imp = imp_data["shap_importance"]
    impurity_imp = imp_data["impurity_importance"]

    rows = []
    for paper_name, column in mapping.items():
        rows.append({
            "Parameter (paper name)": paper_name,
            "Column": column,
            "SHAP importance": round(shap_imp[column], 5),
            "Impurity importance": round(impurity_imp[column], 5),
            "Relief weight": filt_data["relief_weight"][paper_name],
            "ANOVA weight": filt_data["anova_weight"][paper_name],
        })
    df = pd.DataFrame(rows).sort_values("SHAP importance", ascending=False)

    summary_rows = [
        {"Parameter (paper name)": "Spearman r (SHAP vs. impurity)",
         "Column": "", "SHAP importance": imp_data["spearman_r"],
         "Impurity importance": f"p={imp_data['spearman_p']}",
         "Relief weight": "", "ANOVA weight": ""},
        {"Parameter (paper name)": "Spearman r (SHAP vs. Relief)",
         "Column": "", "SHAP importance": filt_data["spearman_shap_vs_relief"]["r"],
         "Impurity importance": f"p={filt_data['spearman_shap_vs_relief']['p']}",
         "Relief weight": "", "ANOVA weight": ""},
        {"Parameter (paper name)": "Spearman r (SHAP vs. ANOVA)",
         "Column": "", "SHAP importance": filt_data["spearman_shap_vs_anova"]["r"],
         "Impurity importance": f"p={filt_data['spearman_shap_vs_anova']['p']}",
         "Relief weight": "", "ANOVA weight": ""},
    ]
    df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
    save_table(df, "tbl_ch4_importance_method_comparison")


def tbl_ch4_rca_summary():
    rows = []
    for split in ["val", "test"]:
        d = _load_json(os.path.join(MODELS_DIR, f"rca_evaluation_{split}.json"))
        rows.append({
            "Split": split, "N non-conforming": d["total_non_conforming"],
            "Tier 1": d["tier_counts"]["tier1_resolved"],
            "Tier 2": d["tier_counts"]["tier2_resolved"],
            "Tier 3": d["tier_counts"]["tier3_escalated"],
            "Resolution rate": d["resolution_rate"],
            "Validator confirmed": d["validator_confirmed"],
            "Validator rate": d["validator_rate"],
        })
    save_table(pd.DataFrame(rows), "tbl_ch4_rca_summary")


def tbl_ch4_counterfactual_case():
    src = os.path.join(DATA_DIR, "counterfactual_case_table.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(f"Required source artifact missing: {src}")
    df = pd.read_csv(src)
    save_table(df, "tbl_ch4_counterfactual_case")


def tbl_ch4_method_comparison():
    rows = []
    for split in ["val", "test"]:
        d = _load_json(os.path.join(MODELS_DIR, f"rca_comparison_{split}.json"))
        for method in ["3-tier RCA", "Greedy"]:
            m = d[method]
            rows.append({
                "Split": split, "Method": method,
                "Resolution rate": m["resolution_rate"],
                "Validator rate": m["validator_rate"],
                "Mean proximity": m["mean_proximity"],
                "Mean sparsity": m["mean_sparsity"],
                "Mean NN distance": m["mean_nn_distance"],
                "McNemar p (vs. other method, same split)": d["mcnemar_test"]["mcnemar_p"],
            })
    save_table(pd.DataFrame(rows), "tbl_ch4_method_comparison")


def tbl_ch4_tier_sensitivity():
    rows = []
    for split in ["val", "test"]:
        d = _load_json(os.path.join(MODELS_DIR, f"tier_sensitivity_{split}.json"))
        for r in d["records"]:
            rows.append({
                "Split": split, "Sweep": r["sweep"],
                "Confidence threshold": r["confidence_threshold"],
                "Max iter": r["max_iter"],
                "Tier 1": r["tier1"], "Tier 2": r["tier2"], "Tier 3": r["tier3"],
                "Resolution rate": r["resolution_rate"],
                "Validator rate": r["validator_rate"],
            })
    save_table(pd.DataFrame(rows), "tbl_ch4_tier_sensitivity")


def tbl_ch4_drift_sensitivity():
    df = pd.read_csv(os.path.join(MODELS_DIR, "drift_validation.csv"))
    pivot = df.pivot(index="feature", columns="k_sigma", values="psi")
    pivot.columns = [f"PSI (k={k})" for k in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"feature": "Feature"})
    save_table(pivot, "tbl_ch4_drift_sensitivity")


def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(MD_DIR, exist_ok=True)
    print("=== Thesis table export ===")

    print("\n-- Chapter 3 --")
    tbl_ch3_feature_specification()
    tbl_ch3_class_distribution()
    tbl_ch3_split_summary()
    tbl_ch3_hyperparameter_provenance()
    tbl_ch3_tree_depth()

    print("\n-- Chapter 4 --")
    tbl_ch4_algorithm_comparison()
    tbl_ch4_test_performance()
    tbl_ch4_per_class_metrics()
    tbl_ch4_confusion_matrix()
    tbl_ch4_baseline_comparison()
    tbl_ch4_shap_global_ranking()
    tbl_ch4_importance_method_comparison()
    tbl_ch4_rca_summary()
    tbl_ch4_counterfactual_case()
    tbl_ch4_method_comparison()
    tbl_ch4_tier_sensitivity()
    tbl_ch4_drift_sensitivity()

    print("\nDone.")


if __name__ == "__main__":
    main()
