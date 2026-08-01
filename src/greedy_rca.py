"""
greedy_rca.py — Greedy counterfactual baseline for comparison with 3-tier RCA.

The greedy approach adjusts all 13 features simultaneously toward the centroid of
all conforming training samples, without SHAP feature selection or LIME direction
guidance. This serves as the ablation baseline in the thesis comparison study:

  3-tier RCA     → SHAP selects (top-5/7) · LIME directs · NN-anchored fallback
  Greedy (this)  → all features · centroid direction · single tier

Same acceptance criterion (P(Acc)+P(Target) ≥ confidence_threshold) and physical
bounds (clip to scaled space [-1, 1]) are applied so comparisons are fair.

Output schema matches CounterfactualRCA.analyze() exactly so compare_rca_methods.py
can treat both interchangeably.
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings

from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings("ignore")

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(BASE_DIR, "models", "checkpoints")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

TARGET_CLASSES       = {1, 2}
CONFIDENCE_THRESHOLD = 0.55
MAX_ITER             = 150
STEP                 = 0.02


class GreedyCounterfactual:
    """
    Single-tier greedy counterfactual: moves all features toward the global
    centroid of conforming training samples until the acceptance threshold is met.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_iter: int = MAX_ITER,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_iter             = max_iter

        self.model         = joblib.load(os.path.join(MODELS_DIR, "current_model.pkl"))
        self.scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        self.feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

        X_train_df  = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
        y_train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
        y_train     = y_train_raw - 1 if y_train_raw.min() > 0 else y_train_raw

        X_train = X_train_df.values
        good_mask = np.isin(y_train, list(TARGET_CLASSES))
        self.centroid = X_train[good_mask].mean(axis=0)
        self.X_train  = X_train
        self.y_train  = y_train

        self.validator = MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=1000,
            random_state=7, early_stopping=True, n_iter_no_change=20
        )
        self.validator.fit(X_train, y_train)

    def _confidence(self, x_scaled):
        proba = self.model.predict_proba(x_scaled.reshape(1, -1))[0]
        return float(proba[1] + proba[2])

    def _is_accepted(self, x_scaled):
        return self._confidence(x_scaled) >= self.confidence_threshold

    def _validate(self, x_scaled):
        pred = int(self.validator.predict(x_scaled.reshape(1, -1))[0])
        return pred in TARGET_CLASSES

    def analyze(self, x_real_df: pd.DataFrame) -> dict:
        """
        Input:  DataFrame (1 row) in real (unscaled) units.
        Output: dict matching CounterfactualRCA.analyze() schema exactly.
        """
        x_scaled = self.scaler.transform(x_real_df)[0]
        proba    = self.model.predict_proba(x_scaled.reshape(1, -1))[0]
        pred     = int(self.model.predict(x_scaled.reshape(1, -1))[0])
        conf     = float(proba[1] + proba[2])

        if self._is_accepted(x_scaled):
            return {
                "tier": 0, "status": "already_acceptable",
                "prediction": pred, "proba": proba.tolist(), "confidence": conf,
                "adjustments": [], "validator_ok": True, "cf_confidence": conf,
                "message": f"Already meets threshold (P{{Acc,Target}}={conf:.1%}).",
            }

        x = x_scaled.copy()
        for _ in range(self.max_iter):
            directions = np.sign(self.centroid - x)
            x += directions * STEP
            x  = np.clip(x, -1.0, 1.0)
            if self._is_accepted(x):
                cf_conf = self._confidence(x)
                vok     = self._validate(x)
                adj     = self._build_adjustments(x_real_df, x)
                return {
                    "tier": 1, "status": "resolved",
                    "prediction": pred, "proba": proba.tolist(), "confidence": conf,
                    "adjustments": adj, "validator_ok": vok,
                    "cf_confidence": round(cf_conf, 4),
                    "message": (f"Greedy (all features → centroid): {len(adj)} adjustments. "
                                f"CF confidence={cf_conf:.1%}. "
                                f"Validator: {'confirmed' if vok else 'unconfirmed'}."),
                }

        return {
            "tier": 3, "status": "escalate",
            "prediction": pred, "proba": proba.tolist(), "confidence": conf,
            "adjustments": [], "validator_ok": False, "cf_confidence": 0.0,
            "message": "Greedy: no counterfactual found within bounds.",
        }

    def _build_adjustments(self, x_real_df, cf_scaled):
        cf_real = self.scaler.inverse_transform(cf_scaled.reshape(1, -1))[0]
        x_real  = x_real_df.values[0]
        result  = []
        for i, fname in enumerate(self.feature_names):
            delta = round(float(cf_real[i]) - float(x_real[i]), 4)
            if abs(delta) > 1e-4:
                result.append({
                    "feature":   fname,
                    "current":   round(float(x_real[i]), 4),
                    "suggested": round(float(cf_real[i]), 4),
                    "delta":     delta,
                    "direction": "↑" if delta > 0 else "↓",
                })
        return sorted(result, key=lambda r: abs(r["delta"]), reverse=True)
