# Draft Captions

Draft captions for every figure and table in `thesis_assets/`, ready for
inclusion with light editing. Each states what is shown, the data split, and
the source artifact, per `MANIFEST.md`. Figure/table numbers are left as
`[Fig. X.Y]` / `[Table X.Y]` placeholders for the author to assign.

---

## Chapter 3 Figures

**fig_ch3_class_distribution** — [Fig. X.Y]. Class-count distribution of the
injection-molding dataset across the training, validation, and test splits.
Bars are grouped by quality class (Waste, Acceptable, Target, Inefficient)
and coloured by split. Source: `thesis_assets/data/class_distribution.json`.

**fig_ch3_correlation_heatmap** — [Fig. X.Y]. Pearson correlation matrix of
the 13 process parameters, computed on the training split (n=870). Annotated
cells give the correlation coefficient to two decimal places. Source:
`thesis_assets/data/correlation_matrix.csv`.

**fig_ch3_feature_ranges** — [Fig. X.Y]. Standardised (z-score) distribution
of each of the 13 process parameters over the full raw dataset (n=1451),
shown as box plots to allow direct comparison of scale and spread across
parameters measured in different physical units. Source: `data/raw/raw_data.csv`.

## Chapter 3 Tables

**tbl_ch3_feature_specification** — [Table X.Y]. The 13 process parameters
with their physical quantity, measurement unit, and descriptive statistics
(min, max, mean, median, standard deviation) computed on the raw, unscaled
dataset. Eight of the thirteen units could not be verified against the
source paper and are marked UNVERIFIED (see final report / `feature_specification.json`).

**tbl_ch3_class_distribution** — [Table X.Y]. Sample counts and percentages
per quality class for the full dataset and for each of the training,
validation, and test splits, including the aggregate conforming
({Acceptable, Target}) and non-conforming ({Waste, Inefficient}) counts.

**tbl_ch3_split_summary** — [Table X.Y]. Sample sizes and proportions for the
60/20/20 train/validation/test split, the random seed (42), and confirmation
that per-class proportions are preserved within 2 percentage points across
splits (stratified split).

**tbl_ch3_hyperparameter_provenance** — [Table X.Y]. Mapping of the champion
Random Forest's hyperparameters against the configuration reported in
Polenta et al. (2022), Table 5, with each deviation classified and cross-
referenced to its justification in `docs/hyperparameter_provenance.md`.

**tbl_ch3_tree_depth** — [Table X.Y]. Configured versus empirically realised
tree depth across the 151 estimators of the champion Random Forest,
demonstrating that the configured `max_depth=79` is not a binding constraint
in practice (realised maximum depth: 20).

## Chapter 4 Figures

**fig_ch4_algorithm_comparison** — [Fig. X.Y]. Five-fold stratified
cross-validation mean accuracy (± one standard deviation) for the six
candidate algorithms evaluated on the training split, ranked by CV mean —
the model-selection criterion used in this work.

**fig_ch4_confusion_matrix_test** — [Fig. X.Y]. Confusion matrix of the
champion model on the held-out test split (n=291), annotated with raw
counts and row-normalised percentages.

**fig_ch4_per_class_vs_baseline** — [Fig. X.Y]. Per-class precision and
recall on this work's held-out test split (n=291) against the Random Forest
row of Polenta et al. (2022), Table 10. **Protocol difference:** the paper's
values are summed across the 5 folds of a stratified 5-fold cross-validation
over the full 1451-sample dataset, not a held-out split; the two are not a
like-for-like comparison and should be read as directionally indicative
only (see `docs/hyperparameter_provenance.md`, Section 5).

**fig_ch4_shap_global_bar** — [Fig. X.Y]. Global feature importance from
SHAP TreeExplainer, computed as the mean absolute SHAP value per feature
over the full training split, aggregated across all four output classes.

**fig_ch4_shap_beeswarm** — [Fig. X.Y]. SHAP summary (beeswarm) plot for the
Target class, computed on a 200-sample random draw from the training split
(seed 42). Colour encodes the underlying feature value; horizontal position
encodes the signed SHAP contribution.

**fig_ch4_shap_local_waterfall** — [Fig. X.Y]. Local SHAP explanations for
three worked test-split cases, one per outcome of interest: a correctly
classified Waste cycle (test index 8), a correctly classified Inefficient
cycle (test index 5), and a correctly classified Target cycle (test index
0). Each panel decomposes the model's predicted-class probability from the
SHAP expected value to the realised prediction, largest contributions first.

**fig_ch4_lime_local** — [Fig. X.Y]. LIME local linear coefficients for the
same three test-split cases as the SHAP waterfall figure above (test indices
8, 5, and 0), for direct SHAP/LIME comparison on identical cycles.

**fig_ch4_importance_method_comparison** — [Fig. X.Y]. Rank comparison of
the 13 process parameters across three importance methods: this work's SHAP
ranking (training split) against the Relief and ANOVA filter rankings
reported in Polenta et al. (2022), Figures 2-3. Spearman rank correlation:
r=0.659 (SHAP vs. Relief), r=0.604 (SHAP vs. ANOVA) — moderate positive
agreement, with "cycle time" ranked first by all three methods.

**fig_ch4_tier_distribution** — [Fig. X.Y]. Distribution of RCA outcomes
(Tier 1, Tier 2, Tier 3, and already-acceptable) across the five confidence
thresholds swept in the sensitivity study, test split (n=147 non-conforming
cycles per threshold).

**fig_ch4_tradeoff_resolution_validator** — [Fig. X.Y]. **Central result
figure.** Resolution rate (falling) and independent-validator agreement rate
(rising) as a function of the acceptance-confidence threshold, test split.
The default operating point (threshold = 0.55) is annotated. Raw counts are
labelled at every marker given the small samples at the tight end of the
sweep — at threshold 0.95, resolution is 7/147 and validator agreement is
6/7.

**fig_ch4_tradeoff_maxiter** — [Fig. X.Y]. The same resolution-rate /
validator-agreement trade-off as a function of the counterfactual search's
iteration budget (10, 25, 50, 150), threshold fixed at 0.55, test split.

**fig_ch4_method_comparison_metrics** — [Fig. X.Y]. Mean proximity, sparsity
(number of adjusted features), and nearest-conforming-neighbour distance for
the 3-tier RCA engine versus the centroid-ablation baseline, test split.

**fig_ch4_validator_by_split_method** — [Fig. X.Y]. Independent-validator
agreement rate for both RCA methods (3-tier, centroid-ablation) on both data
splits (validation, test), with the McNemar exact-test p-value for each
split annotated (val: p=0.625; test: p=0.180 — neither significant at
α=0.05).

**fig_ch4_top_adjusted_features** — [Fig. X.Y]. The five process parameters
most frequently selected for adjustment by the 3-tier RCA engine across all
resolved test-split cases.

**fig_ch4_psi_sensitivity** — [Fig. X.Y]. Population Stability Index (PSI)
for each of the 13 process parameters under synthetic distribution shifts of
magnitude k·σ (k ∈ {0,1,2,3}), with the moderate (PSI=0.10) and critical
(PSI=0.20) drift thresholds marked.

## Chapter 4 Tables

**tbl_ch4_algorithm_comparison** — [Table X.Y]. Five-fold CV mean/std
accuracy and single-split validation accuracy/F1 for all six candidate
algorithms, training split.

**tbl_ch4_test_performance** — [Table X.Y]. Headline test-split (n=291)
performance of the champion model: accuracy, weighted and macro F1, and
non-conformance detection precision/recall.

**tbl_ch4_per_class_metrics** — [Table X.Y]. Per-class precision, recall,
and support on the held-out test split.

**tbl_ch4_confusion_matrix** — [Table X.Y]. Full 4x4 confusion matrix on the
test split with row and column totals.

**tbl_ch4_baseline_comparison** — [Table X.Y]. This work's per-class
precision/recall (test split, n=291) against Polenta et al. (2022), Table
10, Random Forest row. Protocol difference noted in the table itself: the
paper reports samples summed across 5 CV folds over the full dataset, not a
held-out split.

**tbl_ch4_shap_global_ranking** — [Table X.Y]. All 13 process parameters
ranked by mean absolute SHAP value, aggregated across classes, training
split.

**tbl_ch4_importance_method_comparison** — [Table X.Y]. Side-by-side SHAP,
RF impurity, Relief, and ANOVA importance values for all 13 parameters
(mapped to the paper's parameter names), with Spearman rank correlations
between SHAP and each of the other three methods.

**tbl_ch4_rca_summary** — [Table X.Y]. RCA tier counts, resolution rate, and
validator agreement rate on both the validation and test splits at the
default configuration (threshold=0.55, max_iter=150).

**tbl_ch4_counterfactual_case** — [Table X.Y]. The validator-confirmed
worked counterfactual case (test index 39) flattened into a
parameter/original/suggested/delta table, in real engineering units.

**tbl_ch4_method_comparison** — [Table X.Y]. All five comparison metrics
(resolution rate, validator rate, proximity, sparsity, NN distance) for both
RCA methods on both data splits, with the corresponding McNemar p-value.

**tbl_ch4_tier_sensitivity** — [Table X.Y]. All eight sensitivity-sweep
configurations (five confidence thresholds, three additional iteration
budgets) on both the validation and test splits.

**tbl_ch4_drift_sensitivity** — [Table X.Y]. PSI value at each synthetic
shift magnitude (k=0,1,2,3) for each of the 13 process parameters.

---

## Appendix

**figures/appendix/README.md** is a checklist, not a rendered figure — it
lists the 8 application pages that must be captured manually from a running
Streamlit session and is referenced here for completeness rather than
captioned.
