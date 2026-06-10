import shap
import joblib
import pandas as pd
import numpy as np
import os
import warnings
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

CLASS_NAMES = ['Waste', 'Acceptable', 'Target', 'Inefficient']


class XAIEngine:
    def __init__(self):
        print("🧠 Initializing XAI Engine (SHAP + LIME, Differential Mode)...")
        self.model = joblib.load(os.path.join(MODELS_DIR, "current_model.pkl"))
        self.X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
        self.feature_names = list(self.X_train.columns)

        # Model class 2 = original quality 3 (Target). Explain contributions
        # toward Target class so deltas show what prevents a sample reaching it.
        self.target_class_idx = int(np.where(self.model.classes_ == 2)[0][0])

        # SHAP — TreeExplainer for RF/XGB
        self.explainer = shap.TreeExplainer(self.model)

        # LIME — model-agnostic local linear approximation.
        # discretize_continuous=False: use raw feature values (not binned ranges),
        # ensuring feature names in explanations match self.feature_names exactly.
        self.lime_explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.feature_names,
            class_names=CLASS_NAMES,
            mode='classification',
            discretize_continuous=False,
            random_state=42
        )
        print("✅ XAI Engine ready (SHAP TreeExplainer + LIME TabularExplainer).")

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def _get_shap_values(self, row_df):
        """Return 1-D SHAP array (n_features,) for the Target class."""
        sv = np.array(self.explainer.shap_values(row_df))
        if sv.ndim == 3:
            # RF multi-class: (samples, features, classes)
            return sv[0, :, self.target_class_idx]
        elif sv.ndim == 2:
            return sv[0]
        return sv

    def get_differential_diagnosis(self, defect_row, golden_row):
        """
        SHAP delta between a defective sample and a golden reference.
        Returns DataFrame sorted by Delta_Contribution descending.
        """
        shap_defect = self._get_shap_values(defect_row)
        shap_golden = self._get_shap_values(golden_row)
        delta = np.abs(shap_defect - shap_golden)

        return pd.DataFrame({
            'Feature': self.feature_names,
            'Defect_Impact': shap_defect,
            'Golden_Impact': shap_golden,
            'Delta_Contribution': delta
        }).sort_values('Delta_Contribution', ascending=False)

    # ------------------------------------------------------------------
    # LIME
    # ------------------------------------------------------------------

    def get_lime_directions(self, row_df, top_feature_names, label=2, num_samples=2000):
        """
        LIME role: give the DIRECTION of adjustment for a set of features.

        Returns a dict {feature_name: coefficient} for the given label.
        Positive coefficient  → increasing the feature raises P(label).
        Negative coefficient  → decreasing the feature raises P(label).
        Falls back to 0.0 for features LIME did not include (caller should
        use SHAP sign as fallback).
        """
        exp = self.lime_explainer.explain_instance(
            row_df.values[0],
            self.model.predict_proba,
            num_features=len(self.feature_names),
            labels=[label],
            num_samples=num_samples
        )

        # Build name→coefficient map; feature names are returned as-is
        # when discretize_continuous=False.
        raw_list = exp.as_list(label=label)
        coeff_map = {}
        for fname_expr, coeff in raw_list:
            # Match exact name or substring (guard against edge-case formatting)
            for fname in self.feature_names:
                if fname == fname_expr or fname in fname_expr:
                    coeff_map[fname] = float(coeff)
                    break

        return {f: coeff_map.get(f, 0.0) for f in top_feature_names}

    def get_full_lime_explanation(self, row_df, label=2, num_samples=2000):
        """Return full LIME explanation as a sorted DataFrame for display."""
        exp = self.lime_explainer.explain_instance(
            row_df.values[0],
            self.model.predict_proba,
            num_features=len(self.feature_names),
            labels=[label],
            num_samples=num_samples
        )
        rows = []
        for fname_expr, coeff in exp.as_list(label=label):
            matched = next((f for f in self.feature_names
                            if f == fname_expr or f in fname_expr), fname_expr)
            rows.append({'Feature': matched, 'LIME_Coefficient': round(float(coeff), 5)})
        df = pd.DataFrame(rows).sort_values('LIME_Coefficient', key=abs, ascending=False)
        df['Direction'] = df['LIME_Coefficient'].apply(lambda c: '↑ increase' if c > 0 else '↓ decrease')
        return df


if __name__ == "__main__":
    engine = XAIEngine()
    X_val = pd.read_csv(os.path.join(PROCESSED_DIR, "X_val.csv"))
    print("✅ SHAP test:", engine._get_shap_values(X_val.iloc[[0]]).shape)
    lime_d = engine.get_lime_directions(X_val.iloc[[0]], engine.feature_names[:3])
    print("✅ LIME directions (first 3 features):", lime_d)
