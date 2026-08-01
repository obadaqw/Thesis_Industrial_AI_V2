# RF Champion Hyperparameter Provenance

## Reference

Polenta, A.; Tomassini, S.; Falcionelli, N.; Contardo, P.; Dragoni, A. F.; Sernani, P.
"A Comparison of Machine Learning Techniques for the Quality Classification of Molded
Products." *Information*, 2022, 13(6), article 272.
DOI: 10.3390/info13060272

BibTeX key: `Polenta2022`.

---

## What the Paper Reports (Section 3.2.3 and Table 5)

The following values are taken directly from the paper. They are reproduced here
to support the deviation analysis below; nothing in this section is inferred.

- **Best configuration:** 95.04% ± 1.26% mean test accuracy via stratified 5-fold
  cross-validation over all 1,451 samples. Parameters: **151 trees**, **maximum
  depth 79**, **gain ratio** as the splitting criterion.
- **Second-best configuration:** 94.68% accuracy. Parameters: **151 trees**,
  **maximum depth 140**, **information gain** as the splitting criterion.
- **Ensemble method:** Extremely Randomised Trees (ExtraTrees), which selects
  split thresholds randomly rather than greedily optimising the criterion.
- **Features evaluated per split:** int(log(m) + 1), where m is the number of
  features and log is the natural logarithm. For m = 13:
  int(ln(13) + 1) = int(2.565 + 1) = int(3.565) = **3**.
- **Minimal leaf size:** 2. **Minimal size for splitting:** 4.
- **No pruning strategy** was applied.

---

## Hyperparameter Mapping Table

| Hyperparameter | Polenta et al. (2022) | This implementation | Status |
|---|---|---|---|
| `n_estimators` | 151 | 151 | **Identical** |
| `min_samples_leaf` | 2 | 2 | **Identical** |
| `min_samples_split` | 4 | 4 | **Identical** |
| `criterion` | gain ratio (best config) | `'entropy'` (information gain) | **Deviation** — see §1 |
| `max_depth` | 79 (gain ratio row) / 140 (information gain row) | 79 | **Deviation** — see §2 |
| Ensemble type | ExtraTrees | `RandomForestClassifier` | **Deviation** — see §3 |
| Features per split | int(ln(m)+1) = 3 for m=13 | `'sqrt'` → int(√13) = 3 | **Equivalent** — see §4 |
| Evaluation protocol | 5-fold CV, full dataset | Single held-out split | **Deviation** — see §5 |
| `random_state` | Not specified | 42 | **Addition** (reproducibility) |

---

## Deviation Justifications

### §1 — Splitting Criterion

The paper's best-performing configuration uses the **gain ratio** criterion, which
normalises information gain by split entropy to correct for high-cardinality features.
scikit-learn's `RandomForestClassifier` implements only `'gini'` (Gini impurity) and
`'entropy'` (information gain); gain ratio is not available.

`criterion='entropy'` corresponds to information gain, which is the criterion from the
paper's **second-best** row. The paper reports a 0.36 percentage-point difference
between gain ratio (95.04%) and information gain (94.68%) within its own evaluation.
This magnitude represents an upper bound on what is attributable to the criterion
difference under the paper's evaluation protocol.

### §2 — Maximum Depth

Two rows in Table 5 are relevant: the gain ratio row specifies **max_depth = 79**,
and the information gain row specifies **max_depth = 140**. This work uses
`criterion='entropy'` (information gain, §1 above) but `max_depth=79` (the gain
ratio row). The two parameters therefore originate from different rows of Table 5.

**Empirical verification:** The champion model was trained and the realised depth
of each of its 151 estimators measured using `scripts/check_tree_depth.py`. Results
(from `models/tree_depth.json`):

| Statistic | Value |
|---|---|
| Configured max_depth | 79 |
| Realised maximum depth | **20** |
| Realised mean depth | 12.04 |
| Realised median depth | 12.0 |
| Realised minimum depth | 9 |
| Constraint binding? | No |

No tree reaches depth 20, let alone 79 or 140. With 1,451 training samples,
`min_samples_leaf=2`, and `min_samples_split=4`, trees terminate naturally well
below either bound. The discrepancy between 79 and 140 is therefore **empirically
immaterial**: both values produce identical trained models on this dataset.

### §3 — Ensemble Type

The paper uses **Extremely Randomised Trees** (ExtraTrees), in which the split
threshold for each candidate feature is drawn uniformly at random rather than
optimised. `RandomForestClassifier` selects the best threshold among a random
subsample of features, which is a more deterministic search.

`RandomForestClassifier` is used in this work for two reasons:

1. **TreeSHAP compatibility.** The XAI layer (SHAP TreeExplainer) requires a
   model that records the best split threshold per node. ExtraTrees' random-threshold
   design is compatible with TreeSHAP in recent versions, but the entire counterfactual
   RCA chain was developed and validated against `RandomForestClassifier` and tested
   to be correct for that class. Substituting the ensemble type would require
   re-validating every downstream component.
2. **Scikit-learn stack consistency.** All six benchmark algorithms in this work use
   scikit-learn; using `ExtraTreesClassifier` for the champion while keeping
   scikit-learn for the other five would not introduce an inconsistency, but the
   TreeSHAP and reproducibility argument above is the primary justification.

### §4 — Features Evaluated Per Split

The paper specifies int(ln(m) + 1) features per candidate split. scikit-learn's
`max_features='sqrt'` uses int(√m). For **m = 13** (the number of features in this
dataset):

- Paper formula: int(ln(13) + 1) = int(3.565) = **3**
- This work:     int(√13)         = int(3.606) = **3**

The two formulae produce identical values at m = 13. The **Equivalent** classification
applies specifically to this dataset. The equivalence would not hold at other
feature counts (e.g., at m = 7, the paper formula gives 2 and sqrt gives 2;
at m = 20, the paper gives 3 and sqrt gives 4).

### §5 — Evaluation Protocol

The paper reports mean test accuracy from **stratified 5-fold cross-validation over
all 1,451 samples** (95.04% ± 1.26%). This work reports CV accuracy on the training
split only (60% of data, n=870) and a separate held-out test accuracy (n=291, 93.81%).

The two figures are **not directly comparable**. Any thesis statement placing this
work's accuracy alongside the paper's 95.04% must acknowledge that the denominators,
train/test compositions, and averaging procedures differ. The paper's 95.04% is the
reference point for the hyperparameter configuration, not for accuracy comparison.

---

## Champion Selection

The champion model is selected by the highest 5-fold stratified CV mean accuracy
among all six candidate algorithms, as implemented in `get_champion()` in
`src/experiment_tracker.py`, which ranks by `cv_mean` (not single-split `val_acc`).

Results from `models/experiments.json` (six algorithms trained in this work):

| Algorithm | CV mean | CV std | Val acc |
|---|---|---|---|
| RF | **0.9471** | 0.0190 | 0.9310 |
| XGB | 0.9425 | 0.0202 | 0.9414 |
| GB | 0.9379 | 0.0168 | 0.9310 |
| MLP | 0.9218 | 0.0240 | 0.8931 |
| DT | 0.9023 | 0.0170 | 0.8966 |
| KNN | 0.8839 | 0.0043 | 0.8828 |

RF achieves the highest CV mean (0.9471) and is selected as champion. XGB achieves
the highest single-split val_acc (0.9414), which is why the selection criterion is
CV mean: a single held-out split can be misleading.

Held-out test accuracy (RF, n=291): **93.81%** (from `models/test_evaluation.json`).
