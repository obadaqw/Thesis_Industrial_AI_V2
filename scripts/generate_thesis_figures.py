"""
generate_thesis_figures.py — Print-ready figure generation for thesis
Chapters 3-4.

Every figure reads its numbers, at generation time, from an artifact already
on disk (thesis_assets/data/*.json produced by the export_*.py scripts, or
models/*.json produced by the evaluation scripts). No result value is
hardcoded. Where a required source figure input does not exist (the Polenta
et al. 2022 Table 10 per-class baseline, which this repository does not
have), the figure is skipped with a loud printed explanation rather than
fabricated.

Style: serif family, 300 DPI, white background, black text, no top/right
spines, light-grey low-z-order gridlines, colourblind-safe palette, series
additionally distinguished by marker/linestyle/hatch for greyscale legibility.

Usage:
    python scripts/generate_thesis_figures.py
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

DATA_DIR = os.path.join(BASE, "thesis_assets", "data")
MODELS_DIR = os.path.join(BASE, "models")
CKPT = os.path.join(MODELS_DIR, "checkpoints")
PROC = os.path.join(BASE, "data", "processed")
FIG_CH3 = os.path.join(BASE, "thesis_assets", "figures", "ch3")
FIG_CH4 = os.path.join(BASE, "thesis_assets", "figures", "ch4")

CLASS_NAMES = ["Waste", "Acceptable", "Target", "Inefficient"]

# Polenta et al. (2022), Table 10, Random Forest row -- reproduced verbatim
# as a reference constant (percentages converted to fractions). This is the
# paper's own RF result (summed across the 5 test folds of its stratified
# 5-fold CV over all 1451 samples), the natural baseline since RF is also
# this work's champion algorithm. NOT directly comparable to this work's
# single held-out test-split (n=291) evaluation protocol -- see
# docs/hyperparameter_provenance.md Section 5 and the caption/note below.
POLENTA_TABLE10_RF = {
    "Waste":       {"precision": 0.9722, "recall": 0.9459},
    "Acceptable":  {"precision": 0.9511, "recall": 0.9581},
    "Target":      {"precision": 0.9437, "recall": 0.9194},
    "Inefficient": {"precision": 0.9342, "recall": 0.9736},
}

# ── Shared print style ──────────────────────────────────────────────────────
# Okabe-Ito colourblind-safe palette.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]
HATCHES = ["", "//", "xx", "..", "\\\\", "++"]


def series_style(i: int):
    """(color, marker, linestyle) for the i-th of an arbitrary number of line
    series. Naive `PALETTE[i % 8]` / `MARKERS[i % 8]` cycling collides for
    n > 8 since both lists share length 8 -- item 8 would be visually
    identical to item 0. This staggers marker and linestyle across each
    'lap' through the palette so repeats stay distinguishable up to 8*4*8
    series before any two are identical in all three channels."""
    lap = i // len(PALETTE)
    color = PALETTE[i % len(PALETTE)]
    marker = MARKERS[(i % len(PALETTE) + lap) % len(MARKERS)]
    linestyle = LINESTYLES[lap % len(LINESTYLES)]
    return color, marker, linestyle

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,   # embed fonts as TrueType, not paths
})


def savefig(fig, chapter_dir: str, stem: str) -> None:
    png_path = os.path.join(chapter_dir, f"{stem}.png")
    pdf_path = os.path.join(chapter_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}.png / .pdf")


def _load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required source artifact missing: {path}")
    with open(path) as f:
        return json.load(f)


# ── Chapter 3 ────────────────────────────────────────────────────────────────

def fig_ch3_class_distribution():
    d = _load_json(os.path.join(DATA_DIR, "class_distribution.json"))
    splits = ["train", "val", "test"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    for i, split in enumerate(splits):
        counts = [d["splits"][split]["by_class"][c]["count"] for c in CLASS_NAMES]
        ax.bar(x + (i - 1) * width, counts, width, label=split.capitalize(),
               color=PALETTE[i], hatch=HATCHES[i], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Sample count")
    ax.set_xlabel("Quality class")
    ax.legend(title="Split")
    fig.tight_layout()
    savefig(fig, FIG_CH3, "fig_ch3_class_distribution")


def fig_ch3_correlation_heatmap():
    corr = pd.read_csv(os.path.join(DATA_DIR, "correlation_matrix.csv"), index_col=0)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    n = len(corr)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [c if len(c) < 25 else c[:22] + "..." for c in corr.columns]
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(v) > 0.6 else "black")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    savefig(fig, FIG_CH3, "fig_ch3_correlation_heatmap")


def fig_ch3_feature_ranges():
    df_raw = pd.read_csv(os.path.join(BASE, "data", "raw", "raw_data.csv"))
    df_raw.columns = [c.strip() for c in df_raw.columns]
    feature_names = joblib.load(os.path.join(CKPT, "feature_names.pkl"))
    z = (df_raw[feature_names] - df_raw[feature_names].mean()) / df_raw[feature_names].std(ddof=1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot([z[f].values for f in feature_names], vert=False, patch_artist=True,
                     medianprops=dict(color="black"))
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE[0])
        patch.set_alpha(0.5)
        patch.set_edgecolor("black")
    labels = [f if len(f) < 35 else f[:32] + "..." for f in feature_names]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Standardised value (z-score)")
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
    fig.tight_layout()
    savefig(fig, FIG_CH3, "fig_ch3_feature_ranges")


# ── Chapter 4 ────────────────────────────────────────────────────────────────

def fig_ch4_algorithm_comparison():
    with open(os.path.join(MODELS_DIR, "experiments.json")) as f:
        runs = json.load(f)
    df = pd.DataFrame(runs).sort_values("cv_mean")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(df))
    ax.barh(y, df["cv_mean"], xerr=df["cv_std"], color=PALETTE[0],
            edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["algo"])
    ax.set_xlabel("5-fold CV mean accuracy (± std)")
    ax.set_xlim(0.8, 1.0)
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_algorithm_comparison")


def fig_ch4_confusion_matrix_test():
    d = _load_json(os.path.join(MODELS_DIR, "test_evaluation.json"))
    cm = np.array(d["confusion_matrix"])
    row_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i,j]}\n({row_pct[i,j]:.0%})", ha="center", va="center",
                    fontsize=9, color="white" if row_pct[i, j] > 0.5 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row percentage")
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_confusion_matrix_test")


def fig_ch4_per_class_vs_baseline():
    d = _load_json(os.path.join(MODELS_DIR, "test_evaluation.json"))
    this_work = d["per_class"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    for ax, metric in zip(axes, ["precision", "recall"]):
        this_vals = [this_work[c][metric] for c in CLASS_NAMES]
        paper_vals = [POLENTA_TABLE10_RF[c][metric] for c in CLASS_NAMES]
        ax.bar(x - width / 2, this_vals, width, label="This work (test, n=291)",
               color=PALETTE[0], hatch=HATCHES[0], edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, paper_vals, width, label="Polenta et al. (2022), Table 10",
               color=PALETTE[1], hatch=HATCHES[1], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
        ax.set_title(metric.capitalize())
        ax.set_ylim(0.85, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=7, loc="lower left", frameon=True,
                    facecolor="white", framealpha=0.9, edgecolor="black")
    fig.suptitle("Per-class precision/recall: this work vs. Polenta et al. (2022) RF "
                 "-- different evaluation protocols, see caption", fontsize=10)
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_per_class_vs_baseline")


def fig_ch4_shap_global_bar():
    d = _load_json(os.path.join(DATA_DIR, "shap_global_importance.json"))
    imp = d["aggregated_across_classes"]
    items = sorted(imp.items(), key=lambda kv: kv[1])
    feats, vals = zip(*items)
    labels = [f if len(f) < 35 else f[:32] + "..." for f in feats]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(feats)), vals, color=PALETTE[2], edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (aggregated across classes, training split)")
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_shap_global_bar")


def fig_ch4_shap_beeswarm(engine):
    import shap
    X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
    sample = X_train.sample(n=min(200, len(X_train)), random_state=42)
    sv = np.array(engine.explainer.shap_values(sample))
    sv_target = sv[:, :, engine.target_class_idx]  # Target class, consistent
                                                     # with the app's own
                                                     # differential-explanation convention

    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv_target, sample, feature_names=engine.feature_names,
                       show=False, plot_size=None)
    fig = plt.gcf()
    fig.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=8)
    plt.title("SHAP summary (Target class), 200-sample draw from training split")
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_shap_beeswarm")


def _draw_waterfall_panel(ax, base_value: float, contributions: dict, max_display: int = 7):
    """Manually-drawn waterfall (not shap.plots.waterfall: that renderer manages
    its own figure/text placement and bleeds across neighbouring subplots when
    composed into a multi-panel figure -- confirmed by inspection). Rows are
    ordered by |contribution| descending, cumulatively stacked from base_value
    to f(x); mathematically equivalent to SHAP's own ordering convention."""
    items = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = items[:max_display - 1]
    rest = items[max_display - 1:]
    if rest:
        top = top + [(f"{len(rest)} other features", sum(v for _, v in rest))]

    cum = base_value
    rows = []
    for feat, val in top:
        rows.append((feat, cum, val))
        cum += val
    f_x = cum

    y_pos = np.arange(len(rows))[::-1]
    for (feat, start, val), y in zip(rows, y_pos):
        color = PALETTE[1] if val >= 0 else PALETTE[0]
        hatch = HATCHES[1] if val >= 0 else HATCHES[0]
        ax.barh(y, val, left=start, color=color, hatch=hatch,
                edgecolor="black", linewidth=0.5, height=0.6)
        ax.text(start + val, y, f" {val:+.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0][:28] for r in rows], fontsize=7)
    ax.axvline(base_value, color="black", linewidth=0.7, linestyle=":")
    ax.axvline(f_x, color="black", linewidth=0.9, linestyle="-")
    ax.text(base_value, len(rows) - 0.5, f"E[f(x)]={base_value:.3f}",
            fontsize=7, ha="center", va="bottom", rotation=90)
    ax.text(f_x, len(rows) - 0.5, f"f(x)={f_x:.3f}",
            fontsize=7, ha="center", va="bottom", rotation=90)
    ax.set_xlabel("Contribution to P(predicted class)")


def fig_ch4_shap_local_waterfall(engine):
    d = _load_json(os.path.join(DATA_DIR, "local_explanations.json"))
    cases = d["cases"]
    expected = engine.explainer.expected_value

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (name, case) in zip(axes, cases.items()):
        pred = case["predicted_class"]
        _draw_waterfall_panel(ax, float(expected[pred]), case["shap_contribution"])
        ax.set_title(f"{case['predicted_label']} (test idx "
                      f"{case['sample_index_in_test_split']})", fontsize=10)
    fig.suptitle("Local SHAP waterfalls, three worked test-split cases (predicted class)", y=1.02)
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_shap_local_waterfall")


def fig_ch4_lime_local():
    d = _load_json(os.path.join(DATA_DIR, "local_explanations.json"))
    cases = d["cases"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, case) in zip(axes, cases.items()):
        coeffs = case["lime_coefficient"]
        items = sorted(coeffs.items(), key=lambda kv: kv[1])[-8:]
        feats, vals = zip(*items)
        colors = [PALETTE[1] if v < 0 else PALETTE[0] for v in vals]
        ax.barh(range(len(feats)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels([f[:22] for f in feats], fontsize=7)
        ax.set_title(f"{case['predicted_label']}", fontsize=10)
        ax.axvline(0, color="black", linewidth=0.6)
    fig.suptitle("LIME local coefficients, three worked test-split cases")
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_lime_local")


def fig_ch4_importance_method_comparison():
    d = _load_json(os.path.join(DATA_DIR, "shap_vs_filters.json"))
    params = list(d["parameter_mapping"].keys())
    shap_vals = d["shap_importance_by_paper_name"]
    relief_vals = d["relief_weight"]
    anova_vals = d["anova_weight"]

    def _rank(vals_dict):
        ordered = sorted(params, key=lambda p: vals_dict[p], reverse=True)
        return {p: ordered.index(p) + 1 for p in params}

    shap_rank = _rank(shap_vals)
    relief_rank = _rank(relief_vals)
    anova_rank = _rank(anova_vals)

    fig, ax = plt.subplots(figsize=(8, 7))
    xs = [0, 1, 2]
    xlabels = ["SHAP", "Relief", "ANOVA"]
    for i, p in enumerate(params):
        ys = [shap_rank[p], relief_rank[p], anova_rank[p]]
        color, marker, linestyle = series_style(i)
        ax.plot(xs, ys, marker=marker, linestyle=linestyle, color=color,
                label=p, linewidth=1.2, markersize=6)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Rank (1 = most important)")
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_importance_method_comparison")


def _tier_sensitivity_threshold_rows(split="test"):
    d = _load_json(os.path.join(MODELS_DIR, f"tier_sensitivity_{split}.json"))
    return [r for r in d["records"] if r["sweep"] == "threshold"]


def _tier_sensitivity_maxiter_rows(split="test"):
    d = _load_json(os.path.join(MODELS_DIR, f"tier_sensitivity_{split}.json"))
    rows = [r for r in d["records"] if r["sweep"] == "max_iter"]
    # max_iter=150 is only captured under the threshold sweep (thresh=0.55).
    thresh_rows = [r for r in d["records"] if r["sweep"] == "threshold"]
    row_150 = next(r for r in thresh_rows if r["confidence_threshold"] == 0.55)
    rows = rows + [{**row_150, "sweep": "max_iter", "max_iter": 150}]
    return sorted(rows, key=lambda r: r["max_iter"])


def fig_ch4_tier_distribution():
    rows = _tier_sensitivity_threshold_rows("test")
    threshs = [r["confidence_threshold"] for r in rows]
    t1 = np.array([r["tier1"] for r in rows])
    t2 = np.array([r["tier2"] for r in rows])
    t3 = np.array([r["tier3"] for r in rows])
    total = np.array([r["total"] for r in rows])
    other = total - t1 - t2 - t3  # already-acceptable / not attempted

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(threshs))
    bottom = np.zeros(len(threshs))
    for vals, label, color, hatch in [
        (t1, "Tier 1 (SHAP+LIME)", PALETTE[0], HATCHES[0]),
        (t2, "Tier 2 (NN-anchored)", PALETTE[1], HATCHES[1]),
        (t3, "Tier 3 (escalate)", PALETTE[2], HATCHES[2]),
        (other, "Already acceptable (tier 0)", PALETTE[3], HATCHES[3]),
    ]:
        ax.bar(x, vals, bottom=bottom, label=label, color=color, hatch=hatch,
               edgecolor="black", linewidth=0.5)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in threshs])
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Sample count (test split, n=147 non-conforming)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_tier_distribution")


def fig_ch4_tradeoff_resolution_validator():
    rows = _tier_sensitivity_threshold_rows("test")
    rows = sorted(rows, key=lambda r: r["confidence_threshold"])
    threshs = [r["confidence_threshold"] for r in rows]
    res_rate = [r["resolution_rate"] for r in rows]
    val_rate = [r["validator_rate"] for r in rows]
    resolved_n = [r["resolved"] for r in rows]
    total_n = [r["total"] for r in rows]
    validated_n = [r["validator_confirmed"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5.2))
    ax2 = ax1.twinx()

    l1, = ax1.plot(threshs, res_rate, marker="o", linestyle="-", color=PALETTE[0],
                    linewidth=1.8, markersize=7, label="Resolution rate")
    l2, = ax2.plot(threshs, val_rate, marker="s", linestyle="--", color=PALETTE[1],
                    linewidth=1.8, markersize=7, label="Validator agreement rate")

    for x, y, n, tot in zip(threshs, res_rate, resolved_n, total_n):
        ax1.annotate(f"{n}/{tot}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color=PALETTE[0])
    for x, y, n, res in zip(threshs, val_rate, validated_n, resolved_n):
        ax2.annotate(f"{n}/{res}", (x, y), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=8, color=PALETTE[1])

    ax1.axvline(0.55, color="black", linewidth=1.0, linestyle=":")
    ax1.annotate(
        "default operating point\n(threshold = 0.55)",
        xy=(0.55, 1.0), xycoords=("data", "axes fraction"),
        xytext=(25, 12), textcoords="offset points",
        fontsize=8, va="bottom", ha="left",
        arrowprops=dict(arrowstyle="-", color="black", linewidth=0.7),
        annotation_clip=False,
    )
    ax1.legend(handles=[l1, l2], loc="center left", bbox_to_anchor=(0.02, 0.62), fontsize=9)

    ax1.set_xlabel("Confidence threshold: P(Acceptable) + P(Target)")
    ax1.set_ylabel("Resolution rate", color=PALETTE[0])
    ax2.set_ylabel("Validator agreement rate", color=PALETTE[1])
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])
    ax2.tick_params(axis="y", labelcolor=PALETTE[1])
    ax2.grid(False)
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_tradeoff_resolution_validator")


def fig_ch4_tradeoff_maxiter():
    rows = _tier_sensitivity_maxiter_rows("test")
    iters = [r["max_iter"] for r in rows]
    res_rate = [r["resolution_rate"] for r in rows]
    val_rate = [r["validator_rate"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.8))
    ax2 = ax1.twinx()
    l1, = ax1.plot(iters, res_rate, marker="o", linestyle="-", color=PALETTE[0],
                    linewidth=1.8, markersize=7, label="Resolution rate")
    l2, = ax2.plot(iters, val_rate, marker="s", linestyle="--", color=PALETTE[1],
                    linewidth=1.8, markersize=7, label="Validator agreement rate")
    ax1.set_xlabel("Iteration budget (max_iter), threshold = 0.55")
    ax1.set_ylabel("Resolution rate", color=PALETTE[0])
    ax2.set_ylabel("Validator agreement rate", color=PALETTE[1])
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])
    ax2.tick_params(axis="y", labelcolor=PALETTE[1])
    ax2.grid(False)
    ax1.legend(handles=[l1, l2], loc="center right", fontsize=9)
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_tradeoff_maxiter")


def fig_ch4_method_comparison_metrics():
    d = _load_json(os.path.join(MODELS_DIR, "rca_comparison_test.json"))
    metrics = ["mean_proximity", "mean_sparsity", "mean_nn_distance"]
    metric_labels = ["Mean proximity", "Mean sparsity (features)", "Mean NN distance"]
    rca_vals = [d["3-tier RCA"][m] for m in metrics]
    greedy_vals = [d["Greedy"][m] for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, rca_vals, width, label="3-tier RCA", color=PALETTE[0],
           hatch=HATCHES[0], edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, greedy_vals, width, label="Centroid-ablation baseline",
           color=PALETTE[1], hatch=HATCHES[1], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylabel("Value (test split)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_method_comparison_metrics")


def fig_ch4_validator_by_split_method():
    d_val = _load_json(os.path.join(MODELS_DIR, "rca_comparison_val.json"))
    d_test = _load_json(os.path.join(MODELS_DIR, "rca_comparison_test.json"))

    methods = ["3-tier RCA", "Greedy"]
    splits = [("val", d_val), ("test", d_test)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(methods))
    width = 0.35
    for i, (split_name, d) in enumerate(splits):
        vals = [d[m]["validator_rate"] for m in methods]
        ax.bar(x + (i - 0.5) * width, vals, width, label=split_name.capitalize(),
               color=PALETTE[i], hatch=HATCHES[i], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Validator agreement rate")
    ax.set_ylim(0, 1.05)
    for i, (split_name, d) in enumerate(splits):
        p = d["mcnemar_test"]["mcnemar_p"]
        ax.annotate(f"McNemar p={p:.3f}", (i * 0 + 0.5, 0.95 - i * 0.08),
                    fontsize=8, ha="center")
    ax.legend(title="Split")
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_validator_by_split_method")


def fig_ch4_top_adjusted_features():
    d = _load_json(os.path.join(MODELS_DIR, "rca_evaluation_test.json"))
    feats = d["top_adjusted_features"]
    items = sorted(feats.items(), key=lambda kv: kv[1])
    names, counts = zip(*items)
    labels = [f if len(f) < 35 else f[:32] + "..." for f in names]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(range(len(names)), counts, color=PALETTE[4], edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Times selected as a top-adjusted feature (test split)")
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_top_adjusted_features")


def fig_ch4_psi_sensitivity():
    df = pd.read_csv(os.path.join(MODELS_DIR, "drift_validation.csv"))
    features = df["feature"].unique()

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, feat in enumerate(features):
        sub = df[df["feature"] == feat].sort_values("k_sigma")
        color, marker, linestyle = series_style(i)
        ax.plot(sub["k_sigma"], sub["psi"], marker=marker, linestyle=linestyle,
                color=color, linewidth=1.0, markersize=5,
                label=feat[:28], alpha=0.85)
    ax.axhline(0.10, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.20, color="black", linewidth=0.8, linestyle="-")
    x_max = df["k_sigma"].max()
    ax.annotate("critical (PSI=0.20)", xy=(x_max, 0.20), xytext=(6, 8),
                textcoords="offset points", fontsize=7, ha="left")
    ax.annotate("moderate (PSI=0.10)", xy=(x_max, 0.10), xytext=(6, -12),
                textcoords="offset points", fontsize=7, ha="left")
    ax.set_xlabel("Synthetic shift magnitude (k × σ)")
    ax.set_ylabel("Population Stability Index (PSI)")
    ax.legend(fontsize=6, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    savefig(fig, FIG_CH4, "fig_ch4_psi_sensitivity")


def main():
    os.makedirs(FIG_CH3, exist_ok=True)
    os.makedirs(FIG_CH4, exist_ok=True)
    print("=== Thesis figure generation ===")

    print("\n-- Chapter 3 --")
    fig_ch3_class_distribution()
    fig_ch3_correlation_heatmap()
    fig_ch3_feature_ranges()

    print("\n-- Chapter 4 --")
    fig_ch4_algorithm_comparison()
    fig_ch4_confusion_matrix_test()
    fig_ch4_per_class_vs_baseline()

    from xai_engine import XAIEngine
    engine = XAIEngine()

    fig_ch4_shap_global_bar()
    fig_ch4_shap_beeswarm(engine)
    fig_ch4_shap_local_waterfall(engine)
    fig_ch4_lime_local()
    fig_ch4_importance_method_comparison()
    fig_ch4_tier_distribution()
    fig_ch4_tradeoff_resolution_validator()
    fig_ch4_tradeoff_maxiter()
    fig_ch4_method_comparison_metrics()
    fig_ch4_validator_by_split_method()
    fig_ch4_top_adjusted_features()
    fig_ch4_psi_sensitivity()

    print("\nDone.")


if __name__ == "__main__":
    main()
