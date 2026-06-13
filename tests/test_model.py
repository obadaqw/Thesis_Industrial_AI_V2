"""
Tests for the trained Random Forest model — verifies thesis-specified
hyperparameters, class structure, and minimum performance thresholds.
"""
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier


@pytest.mark.unit
class TestRFHyperparameters:
    """The champion RF must match the thesis-tuned specification exactly."""

    def test_is_random_forest(self, model):
        assert isinstance(model, RandomForestClassifier)

    def test_n_estimators(self, model):
        assert model.n_estimators == 151

    def test_max_depth(self, model):
        assert model.max_depth == 79

    def test_criterion(self, model):
        assert model.criterion == "entropy"

    def test_min_samples_leaf(self, model):
        assert model.min_samples_leaf == 2

    def test_min_samples_split(self, model):
        assert model.min_samples_split == 4

    def test_random_state(self, model):
        assert model.random_state == 42


@pytest.mark.unit
class TestModelClasses:
    """Model must produce 4-class output aligned with thesis label encoding."""

    def test_four_classes(self, model):
        assert len(model.classes_) == 4

    def test_class_values(self, model):
        assert np.array_equal(model.classes_, [0, 1, 2, 3])

    def test_target_class_exists(self, model):
        # Model class 2 = original quality 3 (Target) — thesis label shift
        assert 2 in model.classes_

    def test_target_class_index(self, model):
        idx = int(np.where(model.classes_ == 2)[0][0])
        assert idx == 2


@pytest.mark.integration
class TestModelPerformance:
    """Validation accuracy and output sanity on held-out set."""

    def test_val_accuracy_above_threshold(self, model, X_val, y_val):
        acc = model.score(X_val, y_val)
        assert acc >= 0.90, f"Val accuracy {acc:.4f} is below the 90% floor"

    def test_predictions_in_valid_range(self, model, X_val):
        preds = model.predict(X_val)
        assert set(preds).issubset({0, 1, 2, 3})

    def test_predict_proba_sums_to_one(self, model, X_val):
        probas = model.predict_proba(X_val)
        row_sums = probas.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_shape(self, model, X_val):
        probas = model.predict_proba(X_val)
        assert probas.shape == (len(X_val), 4)

    def test_all_classes_predicted(self, model, X_val):
        # A well-generalising model must predict all 4 classes on val set
        preds = model.predict(X_val)
        assert len(np.unique(preds)) == 4, "Model never predicts some classes"
