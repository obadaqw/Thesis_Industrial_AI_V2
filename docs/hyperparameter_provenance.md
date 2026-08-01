# RF Champion Hyperparameter Provenance

**Source:** Polenta, G., Andreis, A., Ferrari, M., Susto, G. A. (2022). "Machine learning
for quality prediction in injection moulding." *Procedia CIRP*, **112**, pp. 401–406.
Table 5 — "Best hyperparameters for Random Forest."

These values are adopted verbatim and are **not** tuned in this work. The thesis
contribution is the XAI + RCA framework applied on top of the reference benchmark model.

## Mapping Table

| Hyperparameter     | Value       | Polenta (2022) Table 5 | Notes                              |
|--------------------|-------------|------------------------|------------------------------------|
| `n_estimators`     | 151         | 151                    | Number of trees                    |
| `max_depth`        | 79          | 79                     | Maximum tree depth                 |
| `criterion`        | `'entropy'` | entropy                | Split quality measure              |
| `min_samples_leaf` | 2           | 2                      | Minimum samples per leaf node      |
| `min_samples_split`| 4           | 4                      | Minimum samples to split a node    |
| `max_features`     | `'sqrt'`    | sqrt                   | Features considered per split      |
| `random_state`     | 42          | —                      | Set for reproducibility (not cited)|

## Champion Selection

The champion model is selected by highest 5-fold stratified CV mean accuracy
across all six candidate algorithms (RF, XGB, GB, MLP, DT, KNN). RF achieves
cv\_mean = 0.9471, which exceeds XGB (0.9425), GB (0.9379), MLP (0.9218),
DT (0.9023), and KNN (0.8839).

Selection criterion: `get_champion()` in `src/experiment_tracker.py` ranks by
`cv_mean` (not single-split `val_acc`), consistent with cross-validation best practice.
