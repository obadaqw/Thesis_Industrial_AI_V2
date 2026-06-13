"""
Tests for XAIEngine — SHAP shape/correctness, LIME directions,
differential diagnosis, and the SHAP+LIME division of roles.
Uses small num_samples for LIME to keep CI fast.
"""
import pytest
import numpy as np
import pandas as pd

N_FEATURES = 13
LIME_FAST  = 200   # small budget for CI speed


@pytest.mark.integration
class TestSHAP:
    def test_shap_output_shape(self, xai_engine, defect_scaled_row):
        sv = xai_engine._get_shap_values(defect_scaled_row)
        assert sv.shape == (N_FEATURES,), f"Expected ({N_FEATURES},), got {sv.shape}"

    def test_shap_all_finite(self, xai_engine, defect_scaled_row):
        sv = xai_engine._get_shap_values(defect_scaled_row)
        assert np.all(np.isfinite(sv)), "SHAP values contain NaN or Inf"

    def test_shap_target_class_index(self, xai_engine):
        assert xai_engine.target_class_idx == 2, \
            "target_class_idx must be 2 (model class 2 = original quality 3 = Target)"

    def test_shap_different_rows_differ(self, xai_engine, defect_scaled_row, good_scaled_row):
        sv_defect = xai_engine._get_shap_values(defect_scaled_row)
        sv_good   = xai_engine._get_shap_values(good_scaled_row)
        assert not np.allclose(sv_defect, sv_good), \
            "SHAP values identical for defect and golden — explainer is broken"


@pytest.mark.integration
class TestDifferentialDiagnosis:
    def test_returns_dataframe(self, xai_engine, defect_scaled_row, good_scaled_row):
        report = xai_engine.get_differential_diagnosis(defect_scaled_row, good_scaled_row)
        assert isinstance(report, pd.DataFrame)

    def test_required_columns(self, xai_engine, defect_scaled_row, good_scaled_row):
        report = xai_engine.get_differential_diagnosis(defect_scaled_row, good_scaled_row)
        for col in ("Feature", "Defect_Impact", "Golden_Impact", "Delta_Contribution"):
            assert col in report.columns

    def test_sorted_descending_by_delta(self, xai_engine, defect_scaled_row, good_scaled_row):
        report = xai_engine.get_differential_diagnosis(defect_scaled_row, good_scaled_row)
        deltas = report["Delta_Contribution"].values
        assert np.all(deltas[:-1] >= deltas[1:]), "Report not sorted by Delta descending"

    def test_all_features_present(self, xai_engine, defect_scaled_row, good_scaled_row):
        report = xai_engine.get_differential_diagnosis(defect_scaled_row, good_scaled_row)
        assert len(report) == N_FEATURES

    def test_delta_is_non_negative(self, xai_engine, defect_scaled_row, good_scaled_row):
        report = xai_engine.get_differential_diagnosis(defect_scaled_row, good_scaled_row)
        assert (report["Delta_Contribution"] >= 0).all()


@pytest.mark.integration
class TestLIMEDirections:
    def test_get_lime_directions_returns_dict(self, xai_engine, defect_scaled_row, feature_names):
        top3 = feature_names[:3]
        result = xai_engine.get_lime_directions(
            defect_scaled_row, top3, label=2, num_samples=LIME_FAST
        )
        assert isinstance(result, dict)

    def test_lime_directions_keys_match_requested(self, xai_engine, defect_scaled_row, feature_names):
        top3 = feature_names[:3]
        result = xai_engine.get_lime_directions(
            defect_scaled_row, top3, label=2, num_samples=LIME_FAST
        )
        assert set(result.keys()) == set(top3)

    def test_lime_direction_values_are_floats(self, xai_engine, defect_scaled_row, feature_names):
        top3 = feature_names[:3]
        result = xai_engine.get_lime_directions(
            defect_scaled_row, top3, label=2, num_samples=LIME_FAST
        )
        for v in result.values():
            assert isinstance(v, float)


@pytest.mark.integration
class TestFullLIMEExplanation:
    def test_returns_dataframe(self, xai_engine, defect_scaled_row):
        df = xai_engine.get_full_lime_explanation(defect_scaled_row, label=2, num_samples=LIME_FAST)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self, xai_engine, defect_scaled_row):
        df = xai_engine.get_full_lime_explanation(defect_scaled_row, label=2, num_samples=LIME_FAST)
        for col in ("Feature", "LIME_Coefficient", "Direction"):
            assert col in df.columns

    def test_direction_values_valid(self, xai_engine, defect_scaled_row):
        df = xai_engine.get_full_lime_explanation(defect_scaled_row, label=2, num_samples=LIME_FAST)
        assert df["Direction"].isin(["↑ increase", "↓ decrease"]).all()

    def test_lime_coefficients_are_finite(self, xai_engine, defect_scaled_row):
        df = xai_engine.get_full_lime_explanation(defect_scaled_row, label=2, num_samples=LIME_FAST)
        assert np.all(np.isfinite(df["LIME_Coefficient"].values))

    def test_all_13_features_returned(self, xai_engine, defect_scaled_row):
        df = xai_engine.get_full_lime_explanation(defect_scaled_row, label=2, num_samples=LIME_FAST)
        assert len(df) == N_FEATURES
