"""
counterfactual_rca.py  —  Layer 3: 3-Tier Counterfactual Root-Cause Analysis

Design decisions implemented:
  #7  SHAP SELECTS which features to adjust (top-k by |SHAP value for Target class|).
      LIME gives the DIRECTION (sign of local linear coefficient for Target class).
  #9  Three-tier strategy in order: T1 → T2 → T3 (escalation).
  #10 ROBUST T2: centroid of ≤15 nearest real good-quality samples; acceptance
      requires P(Acceptable) + P(Target) >= 0.55 (confidence margin).
  #11 Accepted counterfactual: model predicts class ∈ {1, 2}  (Acceptable OR Target).
  #12 Independent MLP validator (random_state=7, family != RF) confirms CF;
      validator_ok flag is returned to the caller/UI.

All counterfactual search is performed in MinMax-scaled space ([-1, 1]).
Physical bounds are enforced via np.clip at every iteration.
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings

from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
TARGET_CLASSES       = {1, 2}   # model 0-based: 1=Acceptable, 2=Target
CONFIDENCE_THRESHOLD = 0.55     # P(Acceptable) + P(Target) to accept CF
T1_TOP_K             = 5        # SHAP selects top-5 features
T2_TOP_K             = 7        # Tier-2 adjusts top-7 features
T2_N_NEIGHBORS       = 15       # centroid of 15 nearest good samples
STEP                 = 0.02     # per-iteration step in scaled [-1,1] space
MAX_ITER             = 150      # max iterations per tier
LIME_SAMPLES         = 2000     # LIME perturbation budget
# ─────────────────────────────────────────────────────────────────────────────


class CounterfactualRCA:

    def __init__(self):
        print("🔬 Initializing Counterfactual RCA (3-Tier Robust)...")

        self.model        = joblib.load(os.path.join(MODELS_DIR, "current_model.pkl"))
        self.scaler       = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        self.feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

        # Scaled training data
        X_train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
        y_train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
        y_train = y_train_raw - 1 if y_train_raw.min() > 0 else y_train_raw

        self.X_train = X_train_df.values
        self.y_train = y_train

        # Good samples for Tier-2 anchor (Acceptable=1 or Target=2 in 0-based)
        good_mask = np.isin(self.y_train, list(TARGET_CLASSES))
        self.good_samples = self.X_train[good_mask]
        print(f"   T2 good-sample pool: {len(self.good_samples)} samples")

        # XAI engine (SHAP + LIME) — imported here to avoid circular deps
        from xai_engine import XAIEngine
        self.xai = XAIEngine()

        # Independent MLP validator — different family (not RF), different seed
        print("   Training independent MLP validator (seed=7)…")
        self.validator = MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=1000,
            random_state=7, early_stopping=True, n_iter_no_change=20
        )
        self.validator.fit(self.X_train, self.y_train)
        print(f"   Validator train acc: {self.validator.score(self.X_train, self.y_train):.4f}")

        print("✅ CounterfactualRCA ready.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _confidence(self, x_scaled):
        """P(Acceptable) + P(Target) for a single scaled vector."""
        proba = self.model.predict_proba(x_scaled.reshape(1, -1))[0]
        return float(proba[1] + proba[2])

    def _is_accepted(self, x_scaled):
        return self._confidence(x_scaled) >= CONFIDENCE_THRESHOLD

    def _validate(self, x_scaled):
        """Independent MLP validator: True if it also predicts {Acceptable, Target}."""
        pred = int(self.validator.predict(x_scaled.reshape(1, -1))[0])
        return pred in TARGET_CLASSES

    def _build_adjustments(self, x_real_df, cf_scaled, feature_indices):
        """Produce list of {feature, current, suggested, delta, direction} dicts."""
        cf_real = self.scaler.inverse_transform(cf_scaled.reshape(1, -1))[0]
        x_real  = x_real_df.values[0]
        result  = []
        for idx in feature_indices:
            fname  = self.feature_names[idx]
            curr   = round(float(x_real[idx]), 4)
            sugg   = round(float(cf_real[idx]), 4)
            delta  = round(sugg - curr, 4)
            result.append({
                "feature":   fname,
                "current":   curr,
                "suggested": sugg,
                "delta":     delta,
                "direction": "↑" if delta > 0 else ("↓" if delta < 0 else "—"),
            })
        return sorted(result, key=lambda r: abs(r["delta"]), reverse=True)

    # ── Tier 1: SHAP selects · LIME directs ──────────────────────────────────

    def _tier1(self, x_scaled):
        x    = x_scaled.copy()
        x_df = pd.DataFrame([x], columns=self.feature_names)

        # SHAP — select top-T1_TOP_K features by |contribution toward Target|
        shap_vals = self.xai._get_shap_values(x_df)
        top_idx   = np.argsort(np.abs(shap_vals))[-T1_TOP_K:]

        # LIME — direction for Target class (label=2)
        top_names = [self.feature_names[i] for i in top_idx]
        try:
            lime_coeffs = self.xai.get_lime_directions(
                x_df, top_names, label=2, num_samples=LIME_SAMPLES
            )
        except Exception:
            lime_coeffs = {}

        # Direction: LIME sign, fallback to SHAP sign
        directions = {}
        for idx in top_idx:
            fname = self.feature_names[idx]
            coeff = lime_coeffs.get(fname, 0.0)
            if coeff == 0.0:
                coeff = float(shap_vals[idx])
            directions[idx] = float(np.sign(coeff)) if coeff != 0 else 1.0

        for _ in range(MAX_ITER):
            for idx, d in directions.items():
                x[idx] += d * STEP
            x = np.clip(x, -1.0, 1.0)
            if self._is_accepted(x):
                return {"success": True, "cf_scaled": x,
                        "feature_indices": top_idx.tolist()}

        return {"success": False}

    # ── Tier 2: Nearest-neighbour anchored ───────────────────────────────────

    def _tier2(self, x_scaled):
        # 15 nearest good-quality samples → centroid
        dists    = euclidean_distances([x_scaled], self.good_samples)[0]
        nn_idx   = np.argsort(dists)[:T2_N_NEIGHBORS]
        centroid = self.good_samples[nn_idx].mean(axis=0)

        # Top-7 features with largest gap to centroid
        gap     = np.abs(centroid - x_scaled)
        top_idx = np.argsort(gap)[-T2_TOP_K:]

        x = x_scaled.copy()
        for _ in range(MAX_ITER):
            for idx in top_idx:
                d = np.sign(centroid[idx] - x[idx])
                x[idx] += d * STEP
            x = np.clip(x, -1.0, 1.0)
            if self._is_accepted(x):
                return {"success": True, "cf_scaled": x,
                        "feature_indices": top_idx.tolist()}

        return {"success": False}

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, x_real_df):
        """
        Input:  DataFrame (1 row) with real (unscaled) sensor values.
        Output: dict with keys:
            tier          int   0=already good, 1=T1 resolved, 2=T2 resolved, 3=escalate
            status        str   'already_acceptable' | 'resolved' | 'escalate'
            prediction    int   model class 0-3 for the INPUT sample
            proba         list  full probability vector for the INPUT sample
            confidence    float P(Acceptable)+P(Target) for the INPUT sample
            adjustments   list  [{feature, current, suggested, delta, direction}]
            validator_ok  bool  independent MLP confirms counterfactual
            message       str   human-readable summary
        """
        x_scaled = self.scaler.transform(x_real_df)[0]
        proba    = self.model.predict_proba(x_scaled.reshape(1, -1))[0]
        pred     = int(self.model.predict(x_scaled.reshape(1, -1))[0])
        conf     = float(proba[1] + proba[2])

        # ── Tier 0: already acceptable ────────────────────────────────────
        if self._is_accepted(x_scaled):
            return {
                "tier": 0, "status": "already_acceptable",
                "prediction": pred, "proba": proba.tolist(), "confidence": conf,
                "adjustments": [], "validator_ok": True,
                "message": (f"Part meets quality threshold "
                            f"(P{{Acc,Target}}={conf:.1%}).")
            }

        # ── Tier 1 ────────────────────────────────────────────────────────
        r1 = self._tier1(x_scaled)
        if r1["success"]:
            adj = self._build_adjustments(x_real_df, r1["cf_scaled"],
                                          r1["feature_indices"])
            vok = self._validate(r1["cf_scaled"])
            cf_conf = self._confidence(r1["cf_scaled"])
            return {
                "tier": 1, "status": "resolved",
                "prediction": pred, "proba": proba.tolist(), "confidence": conf,
                "adjustments": adj, "validator_ok": vok,
                "cf_confidence": round(cf_conf, 4),
                "message": (f"Tier 1 (SHAP+LIME): {T1_TOP_K} adjustments found. "
                            f"CF confidence={cf_conf:.1%}. "
                            f"Validator: {'✅ Confirmed' if vok else '⚠️ Unconfirmed'}.")
            }

        # ── Tier 2 ────────────────────────────────────────────────────────
        r2 = self._tier2(x_scaled)
        if r2["success"]:
            adj = self._build_adjustments(x_real_df, r2["cf_scaled"],
                                          r2["feature_indices"])
            vok = self._validate(r2["cf_scaled"])
            cf_conf = self._confidence(r2["cf_scaled"])
            return {
                "tier": 2, "status": "resolved",
                "prediction": pred, "proba": proba.tolist(), "confidence": conf,
                "adjustments": adj, "validator_ok": vok,
                "cf_confidence": round(cf_conf, 4),
                "message": (f"Tier 2 (NN-Anchored, {T2_N_NEIGHBORS} neighbors): "
                            f"{T2_TOP_K} adjustments found. "
                            f"CF confidence={cf_conf:.1%}. "
                            f"Validator: {'✅ Confirmed' if vok else '⚠️ Unconfirmed'}.")
            }

        # ── Tier 3: escalation ────────────────────────────────────────────
        return {
            "tier": 3, "status": "escalate",
            "prediction": pred, "proba": proba.tolist(), "confidence": conf,
            "adjustments": [], "validator_ok": False, "cf_confidence": 0.0,
            "message": ("No parameter adjustment can resolve this defect within "
                        "physical bounds. Structural root cause — "
                        "manual inspection required.")
        }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    rca = CounterfactualRCA()

    scaler       = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    X_val_scaled  = pd.read_csv(os.path.join(PROCESSED_DIR, "X_val.csv"))
    y_val_raw     = pd.read_csv(os.path.join(PROCESSED_DIR, "y_val.csv")).values.ravel()
    y_val = y_val_raw - 1 if y_val_raw.min() > 0 else y_val_raw

    # Pick first defect sample (not already Acceptable/Target)
    defect_idx = int(np.where(~np.isin(y_val, list(TARGET_CLASSES)))[0][0])
    sample_real = pd.DataFrame(
        scaler.inverse_transform(X_val_scaled.iloc[[defect_idx]]),
        columns=feature_names
    )

    print(f"\n🧪 Analyzing defect sample (original quality {y_val[defect_idx]+1})...")
    result = rca.analyze(sample_real)

    print(f"\nTier:       {result['tier']}")
    print(f"Status:     {result['status']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Message:    {result['message']}")
    if result['adjustments']:
        print("\nAdjustments:")
        for a in result['adjustments']:
            print(f"  {a['direction']} {a['feature']}: {a['current']} → {a['suggested']} (Δ={a['delta']})")
    print(f"\nValidator OK: {result['validator_ok']}")
