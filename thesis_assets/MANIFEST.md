# Thesis Asset Manifest

One row per asset produced under `thesis_assets/`. `Proposed number` is left
blank for the author to fill in during writing (e.g. "Fig. 4.7", "Table 3.2").

Regenerate everything with `bash scripts/build_thesis_assets.sh`.
Note: `MANIFEST.md`, `CAPTIONS.md`, and `figures/appendix/README.md` are
hand-authored, not script-generated, and are NOT recreated by
`build_thesis_assets.sh` — do not delete them when regenerating.

---

## Chapter 3 — Dataset and Preprocessing

### Figures

| Filename | Type | Section | Description | Script | Source | Split | Proposed # |
|---|---|---|---|---|---|---|---|
| `figures/ch3/fig_ch3_class_distribution.{png,pdf}` | Figure | 3.x Dataset | Grouped bar of class counts across train/val/test | `generate_thesis_figures.py` | `thesis_assets/data/class_distribution.json` | all | |
| `figures/ch3/fig_ch3_correlation_heatmap.{png,pdf}` | Figure | 3.x Dataset | 13x13 annotated Pearson correlation heatmap | `generate_thesis_figures.py` | `thesis_assets/data/correlation_matrix.csv` | train | |
| `figures/ch3/fig_ch3_feature_ranges.{png,pdf}` | Figure | 3.x Dataset | Standardised box plot of the 13 process parameters | `generate_thesis_figures.py` | `data/raw/raw_data.csv` | full | |

### Tables

| Filename | Type | Section | Description | Script | Source | Split | Proposed # |
|---|---|---|---|---|---|---|---|
| `tbl_ch3_feature_specification.{csv,md}` | Table | 3.x Dataset | 13 parameters: unit, min/max/mean/median/std | `export_thesis_tables.py` | `thesis_assets/data/feature_specification.json` | full (raw) | |
| `tbl_ch3_class_distribution.{csv,md}` | Table | 3.x Dataset | Class counts/pct, full dataset + per split | `export_thesis_tables.py` | `thesis_assets/data/class_distribution.json` | all | |
| `tbl_ch3_split_summary.{csv,md}` | Table | 3.x Dataset | Split sizes, proportions, seed, stratification check | `export_thesis_tables.py` | `thesis_assets/data/split_summary.json` | all | |
| `tbl_ch3_hyperparameter_provenance.{csv,md}` | Table | 3.x Model | RF hyperparameter mapping vs. Polenta et al. (2022) | `export_thesis_tables.py` | `docs/hyperparameter_provenance.md` | n/a | |
| `tbl_ch3_tree_depth.{csv,md}` | Table | 3.x Model | Configured vs. realised RF tree depth | `export_thesis_tables.py` | `models/tree_depth.json` | train | |

---

## Chapter 4 — Results

### Figures

| Filename | Type | Section | Description | Script | Source | Split | Proposed # |
|---|---|---|---|---|---|---|---|
| `figures/ch4/fig_ch4_algorithm_comparison.{png,pdf}` | Figure | 4.1 Model selection | 6-algorithm CV mean ± std, horizontal bar | `generate_thesis_figures.py` | `models/experiments.json` | train (CV) | |
| `figures/ch4/fig_ch4_confusion_matrix_test.{png,pdf}` | Figure | 4.1 Model selection | 4x4 confusion matrix, counts + row % | `generate_thesis_figures.py` | `models/test_evaluation.json` | test | |
| `figures/ch4/fig_ch4_per_class_vs_baseline.{png,pdf}` | Figure | 4.1 Model selection | Per-class precision/recall vs. Polenta et al. (2022) Table 10, RF row (protocol difference noted in caption) | `generate_thesis_figures.py` | `models/test_evaluation.json` + `POLENTA_TABLE10_RF` constant | test vs. paper 5-fold CV | |
| `figures/ch4/fig_ch4_shap_global_bar.{png,pdf}` | Figure | 4.2 XAI | Mean \|SHAP\| per feature, aggregated across classes | `generate_thesis_figures.py` | `thesis_assets/data/shap_global_importance.json` | train | |
| `figures/ch4/fig_ch4_shap_beeswarm.{png,pdf}` | Figure | 4.2 XAI | SHAP summary beeswarm, Target class, 200-sample draw | `generate_thesis_figures.py` | `XAIEngine` (live SHAP call) | train (sample) | |
| `figures/ch4/fig_ch4_shap_local_waterfall.{png,pdf}` | Figure | 4.2 XAI | 3-panel local SHAP waterfall (Waste/Inefficient/Target) | `generate_thesis_figures.py` | `thesis_assets/data/local_explanations.json` | test | |
| `figures/ch4/fig_ch4_lime_local.{png,pdf}` | Figure | 4.2 XAI | LIME coefficient bars, same 3 cases | `generate_thesis_figures.py` | `thesis_assets/data/local_explanations.json` | test | |
| `figures/ch4/fig_ch4_importance_method_comparison.{png,pdf}` | Figure | 4.2 XAI | SHAP vs. Relief vs. ANOVA rank slope chart | `generate_thesis_figures.py` | `thesis_assets/data/shap_vs_filters.json` | train / paper | |
| `figures/ch4/fig_ch4_tier_distribution.{png,pdf}` | Figure | 4.3 RCA | Stacked T1/T2/T3/already-acceptable bars across thresholds | `generate_thesis_figures.py` | `models/tier_sensitivity_test.json` | test | |
| `figures/ch4/fig_ch4_tradeoff_resolution_validator.{png,pdf}` | **Figure (central)** | 4.3 RCA | Resolution rate vs. validator agreement, default point annotated, raw counts labelled | `generate_thesis_figures.py` | `models/tier_sensitivity_test.json` | test | |
| `figures/ch4/fig_ch4_tradeoff_maxiter.{png,pdf}` | Figure | 4.3 RCA | Same trade-off vs. iteration budget | `generate_thesis_figures.py` | `models/tier_sensitivity_test.json` | test | |
| `figures/ch4/fig_ch4_method_comparison_metrics.{png,pdf}` | Figure | 4.3 RCA | 3-tier vs. ablation: proximity/sparsity/NN distance | `generate_thesis_figures.py` | `models/rca_comparison_test.json` | test | |
| `figures/ch4/fig_ch4_validator_by_split_method.{png,pdf}` | Figure | 4.3 RCA | Validator rate, both splits x both methods, McNemar p | `generate_thesis_figures.py` | `models/rca_comparison_{val,test}.json` | val + test | |
| `figures/ch4/fig_ch4_top_adjusted_features.{png,pdf}` | Figure | 4.3 RCA | Adjustment frequency, top-5 features | `generate_thesis_figures.py` | `models/rca_evaluation_test.json` | test | |
| `figures/ch4/fig_ch4_psi_sensitivity.{png,pdf}` | Figure | 4.4 Drift | PSI vs. synthetic shift magnitude, 13 features | `generate_thesis_figures.py` | `models/drift_validation.csv` | val (shifted) | |

### Tables

| Filename | Type | Section | Description | Script | Source | Split | Proposed # |
|---|---|---|---|---|---|---|---|
| `tbl_ch4_algorithm_comparison.{csv,md}` | Table | 4.1 Model selection | 6 algorithms: CV mean/std, val acc/F1 | `export_thesis_tables.py` | `models/experiments.json` | train (CV) + val | |
| `tbl_ch4_test_performance.{csv,md}` | Table | 4.1 Model selection | Headline test-set metrics | `export_thesis_tables.py` | `models/test_evaluation.json` | test | |
| `tbl_ch4_per_class_metrics.{csv,md}` | Table | 4.1 Model selection | Precision/recall/support per class | `export_thesis_tables.py` | `models/test_evaluation.json` | test | |
| `tbl_ch4_confusion_matrix.{csv,md}` | Table | 4.1 Model selection | 4x4 confusion matrix with row/column totals | `export_thesis_tables.py` | `models/test_evaluation.json` | test | |
| `tbl_ch4_baseline_comparison.{csv,md}` | Table | 4.1 Model selection | This work vs. Polenta et al. RF, per class, with protocol-difference note | `export_thesis_tables.py` | `models/test_evaluation.json` + `POLENTA_TABLE10_RF` constant | test vs. paper 5-fold CV | |
| `tbl_ch4_shap_global_ranking.{csv,md}` | Table | 4.2 XAI | Features ranked by mean \|SHAP\| | `export_thesis_tables.py` | `thesis_assets/data/shap_global_importance.json` | train | |
| `tbl_ch4_importance_method_comparison.{csv,md}` | Table | 4.2 XAI | SHAP/impurity/Relief/ANOVA ranks + Spearman r | `export_thesis_tables.py` | `thesis_assets/data/shap_vs_impurity.json`, `shap_vs_filters.json` | train / paper | |
| `tbl_ch4_rca_summary.{csv,md}` | Table | 4.3 RCA | Resolution, tier counts, validator rate, both splits | `export_thesis_tables.py` | `models/rca_evaluation_{val,test}.json` | val + test | |
| `tbl_ch4_counterfactual_case.{csv,md}` | Table | 4.3 RCA | Confirmed worked case, engineering units | `export_thesis_tables.py` | `thesis_assets/data/counterfactual_case_table.csv` | test | |
| `tbl_ch4_method_comparison.{csv,md}` | Table | 4.3 RCA | All 5 metrics, both methods, both splits, McNemar p | `export_thesis_tables.py` | `models/rca_comparison_{val,test}.json` | val + test | |
| `tbl_ch4_tier_sensitivity.{csv,md}` | Table | 4.3 RCA | All 8 sweep configs, both splits | `export_thesis_tables.py` | `models/tier_sensitivity_{val,test}.json` | val + test | |
| `tbl_ch4_drift_sensitivity.{csv,md}` | Table | 4.4 Drift | PSI at each k-sigma per feature | `export_thesis_tables.py` | `models/drift_validation.csv` | val (shifted) | |

---

## Supporting data exhibits (`thesis_assets/data/`)

Not each of these renders as its own figure/table; several feed the figures
and tables above, and the rest are raw exhibits to cite directly in the
Chapter 4/5 narrative text (e.g. "LIME's top-5 features were stable across
all 10 re-runs; see `lime_stability.json`").

| Filename | Feeds | Description | Script | Split |
|---|---|---|---|---|
| `feature_specification.json` | Ch3 table/fig | Per-feature units + descriptive stats | `export_dataset_stats.py` | raw |
| `class_distribution.json` | Ch3 table/fig | Class counts/pct, full + per split | `export_dataset_stats.py` | all |
| `correlation_matrix.csv` | Ch3 fig | 13x13 Pearson correlation | `export_dataset_stats.py` | train |
| `split_summary.json` | Ch3 table | Split sizes/proportions/seed | `export_dataset_stats.py` | all |
| `shap_global_importance.json` | Ch4 table/fig | Mean \|SHAP\|, aggregated + per class | `export_xai_results.py` | train |
| `shap_vs_impurity.json` | Ch4 table | SHAP vs. RF impurity importance, Spearman r | `export_xai_results.py` | train |
| `shap_vs_filters.json` | Ch4 table/fig | SHAP vs. Relief/ANOVA (Polenta 2022 Figs 2-3) | `export_xai_results.py` | train / paper |
| `local_explanations.json` | Ch4 fig | 3 worked cases: raw values, proba, SHAP, LIME | `export_xai_results.py` | test |
| `shap_lime_agreement.json` | Narrative (4.2) | SHAP-LIME rank agreement, 50 test samples | `export_xai_results.py` | test |
| `lime_stability.json` | Narrative (4.2) | LIME repeatability, 10 runs on 1 fixed sample | `export_xai_results.py` | test |
| `xai_timing.json` | Narrative (4.2) | SHAP/LIME wall-clock timing, cache hit rate | `export_xai_results.py` | test |
| `counterfactual_cases.json` | Ch4 table | 3 worked RCA cases (confirmed/unconfirmed/tier-0) | `export_rca_cases.py` | test |
| `counterfactual_case_table.csv` | Ch4 table | Confirmed case, flattened | `export_rca_cases.py` | test |
| `unresolved_analysis.json` | Narrative (4.3) | 7 "unresolved" vs. 140 resolved cases compared | `export_rca_cases.py` | test |
| `process_capability.json` | Narrative (Ch5) | Cp/Cpk per feature (limits ASSUMED, see file) | `export_quality_modules.py` | full (raw) |
| `oee_simulation.json` | Narrative (Ch5) | One representative OEE scenario (Digital Twin) | `export_quality_modules.py` | test (replayed) |
| `traceability_sample.json` | Narrative (Ch5) | 5 demonstration audit-trail records (see provenance note in file) | `export_quality_modules.py` | test |
| `llm_report_sample.md` | Narrative (Ch5) | 1 verbatim Groq/Llama-3.3 shift-report exhibit (sample idx 39, same cycle as `tbl_ch4_counterfactual_case`) | `export_quality_modules.py` | test |

---

## Appendix

| Filename | Type | Description |
|---|---|---|
| `figures/appendix/README.md` | Instructions | List of 8 application pages to screenshot manually; cannot be produced by a script |

---

## Provenance note: Polenta et al. (2022) Table 10

`fig_ch4_per_class_vs_baseline` and `tbl_ch4_baseline_comparison` compare
this work against the **Random Forest row of Polenta et al. (2022), Table
10**, supplied by the author as an image and transcribed verbatim into the
`POLENTA_TABLE10_RF` constant in both `scripts/generate_thesis_figures.py`
and `scripts/export_thesis_tables.py`. The paper's own evaluation protocol
(samples summed across 5 folds of stratified CV over the full 1451-sample
dataset) differs from this work's single held-out test split (n=291); both
outputs carry that caveat directly in their caption/note. If the RF row
should be replaced or extended with other algorithms from Table 10, edit the
`POLENTA_TABLE10_RF` dict in both scripts and re-run
`scripts/build_thesis_assets.sh`.
