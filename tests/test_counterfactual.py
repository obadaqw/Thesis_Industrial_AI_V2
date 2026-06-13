"""
Tests for CounterfactualRCA — 3-tier logic, result schema, validator flag,
and threshold enforcement. Session-scoped fixtures keep training overhead to once.
"""
import pytest
import numpy as np

REQUIRED_KEYS = {
    "tier", "status", "prediction", "proba",
    "confidence", "adjustments", "validator_ok", "message"
}
ADJUSTMENT_KEYS = {"feature", "current", "suggested", "delta", "direction"}
TARGET_CLASSES   = {1, 2}
CONFIDENCE_THRESHOLD = 0.55


@pytest.mark.integration
class TestResultSchema:
    def test_good_sample_tier0(self, cf_rca, good_row):
        result = cf_rca.analyze(good_row)
        assert result["tier"]   == 0
        assert result["status"] == "already_acceptable"

    def test_good_sample_required_keys(self, cf_rca, good_row):
        result = cf_rca.analyze(good_row)
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_defect_sample_required_keys(self, cf_rca, defect_row):
        result = cf_rca.analyze(defect_row)
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_tier_is_int_0_to_3(self, cf_rca, defect_row):
        result = cf_rca.analyze(defect_row)
        assert result["tier"] in {0, 1, 2, 3}

    def test_status_is_valid_string(self, cf_rca, defect_row):
        result = cf_rca.analyze(defect_row)
        assert result["status"] in {"already_acceptable", "resolved", "escalate"}

    def test_proba_sums_to_one(self, cf_rca, defect_row):
        result = cf_rca.analyze(defect_row)
        assert sum(result["proba"]) == pytest.approx(1.0, abs=1e-5)

    def test_proba_length_is_4(self, cf_rca, defect_row):
        result = cf_rca.analyze(defect_row)
        assert len(result["proba"]) == 4

    def test_confidence_in_unit_interval(self, cf_rca, defect_row):
        result = cf_rca.analyze(defect_row)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_validator_ok_is_bool(self, cf_rca, defect_row):
        result = cf_rca.analyze(defect_row)
        assert isinstance(result["validator_ok"], bool)

    def test_message_is_nonempty_string(self, cf_rca, defect_row):
        result = cf_rca.analyze(defect_row)
        assert isinstance(result["message"], str) and len(result["message"]) > 0


@pytest.mark.integration
class TestTier0Logic:
    def test_good_sample_no_adjustments(self, cf_rca, good_row):
        result = cf_rca.analyze(good_row)
        assert result["adjustments"] == []

    def test_good_sample_validator_ok(self, cf_rca, good_row):
        result = cf_rca.analyze(good_row)
        assert result["validator_ok"] is True

    def test_good_sample_confidence_above_threshold(self, cf_rca, good_row):
        result = cf_rca.analyze(good_row)
        assert result["confidence"] >= CONFIDENCE_THRESHOLD


@pytest.mark.integration
class TestAdjustmentSchema:
    def test_adjustments_have_required_keys(self, cf_rca, defect_row):
        result = cf_rca.analyze(defect_row)
        for adj in result["adjustments"]:
            assert ADJUSTMENT_KEYS.issubset(adj.keys()), \
                f"Adjustment missing keys: {ADJUSTMENT_KEYS - adj.keys()}"

    def test_direction_arrow_values(self, cf_rca, defect_row):
        result = cf_rca.analyze(defect_row)
        for adj in result["adjustments"]:
            assert adj["direction"] in {"↑", "↓", "—"}

    def test_delta_matches_suggested_minus_current(self, cf_rca, defect_row):
        result = cf_rca.analyze(defect_row)
        for adj in result["adjustments"]:
            expected_delta = round(adj["suggested"] - adj["current"], 4)
            assert adj["delta"] == pytest.approx(expected_delta, abs=0.001)

    def test_tier3_has_no_adjustments(self, cf_rca, defect_row):
        result = cf_rca.analyze(defect_row)
        if result["tier"] == 3:
            assert result["adjustments"] == []


@pytest.mark.integration
class TestConfidenceThreshold:
    def test_threshold_constant(self, cf_rca):
        from counterfactual_rca import CONFIDENCE_THRESHOLD as CF_THRESH
        assert CF_THRESH == pytest.approx(0.55)

    def test_tier0_confidence_above_threshold(self, cf_rca, good_row):
        result = cf_rca.analyze(good_row)
        # Tier 0 means the input already meets threshold
        assert result["confidence"] >= CONFIDENCE_THRESHOLD


@pytest.mark.integration
class TestValidatorIndependence:
    def test_validator_is_mlp(self, cf_rca):
        from sklearn.neural_network import MLPClassifier
        assert isinstance(cf_rca.validator, MLPClassifier)

    def test_validator_seed_differs_from_champion(self, cf_rca):
        # Validator must use seed != 42 to be truly independent
        assert cf_rca.validator.random_state != 42
        assert cf_rca.validator.random_state == 7
