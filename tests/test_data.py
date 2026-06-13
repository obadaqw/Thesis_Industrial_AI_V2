"""
Data integrity tests — verify pipeline outputs match thesis specification.
"""
import os
import pytest
import numpy as np
import pandas as pd
import joblib
import yaml

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, "models", "checkpoints")
PROC_DIR   = os.path.join(BASE, "data", "processed")
GOLDEN_DIR = os.path.join(BASE, "data", "golden_store")
CONFIG_DIR = os.path.join(BASE, "config")

N_FEATURES = 13
EXPECTED_FEATURES = [
    "Melt temperature", "Mold temperature", "time_to_fill",
    "ZDx - Plasticizing time", "ZUx - Cycle time", "SKx - Closing force",
    "SKs - Clamping force peak value", "Ms - Torque peak value current cycle",
    "Mm - Torque mean value current cycle",
    "APSs - Specific back pressure peak value",
    "APVs - Specific injection pressure peak value",
    "CPn - Screw position at the end of hold pressure", "SVo - Shot volume",
]


@pytest.mark.unit
class TestScaler:
    def test_scaler_type(self, scaler):
        from sklearn.preprocessing import MinMaxScaler
        assert isinstance(scaler, MinMaxScaler), "Scaler must be MinMaxScaler (thesis spec)"

    def test_scaler_feature_range(self, scaler):
        assert scaler.feature_range == (-1, 1), "Feature range must be (-1, 1)"

    def test_scaler_fitted(self, scaler):
        assert hasattr(scaler, "data_min_"), "Scaler must be fitted"
        assert scaler.n_features_in_ == N_FEATURES


@pytest.mark.unit
class TestTrainStats:
    def test_train_stats_keys(self, train_stats):
        assert "mean" in train_stats, "train_stats.pkl must have 'mean' key"
        assert "std" in train_stats,  "train_stats.pkl must have 'std' key"

    def test_train_stats_length(self, train_stats):
        assert len(train_stats["mean"]) == N_FEATURES
        assert len(train_stats["std"])  == N_FEATURES

    def test_train_stats_positive_std(self, train_stats):
        assert (train_stats["std"] > 0).all(), "All std values must be positive"


@pytest.mark.unit
class TestFeatureNames:
    def test_feature_count(self, feature_names):
        assert len(feature_names) == N_FEATURES

    def test_feature_names_match_spec(self, feature_names):
        assert set(feature_names) == set(EXPECTED_FEATURES)


@pytest.mark.integration
class TestProcessedData:
    def test_X_train_shape(self, X_train):
        assert X_train.shape[1] == N_FEATURES
        assert len(X_train) > 0

    def test_X_val_shape(self, X_val):
        assert X_val.shape[1] == N_FEATURES
        assert len(X_val) > 0

    def test_X_train_scaled_range(self, X_train):
        # MinMaxScaler fit on train → training data must be exactly in [-1, 1]
        assert X_train.values.min() >= -1.0 - 1e-6
        assert X_train.values.max() <=  1.0 + 1e-6

    def test_y_train_classes(self, y_train):
        assert set(np.unique(y_train)).issubset({0, 1, 2, 3})

    def test_y_val_classes(self, y_val):
        assert set(np.unique(y_val)).issubset({0, 1, 2, 3})

    def test_stratified_split_all_classes(self, y_train, y_val):
        # Stratified split must preserve all 4 classes in both sets
        assert len(np.unique(y_train)) == 4, "Training set must contain all 4 classes"
        assert len(np.unique(y_val))   == 4, "Validation set must contain all 4 classes"


@pytest.mark.unit
class TestGoldenStore:
    def test_golden_file_exists(self):
        assert os.path.exists(os.path.join(GOLDEN_DIR, "golden_samples.csv"))

    def test_golden_shape(self):
        df = pd.read_csv(os.path.join(GOLDEN_DIR, "golden_samples.csv"))
        assert df.shape[1] == N_FEATURES, "Golden store must have 13 features"
        assert len(df) >= 50, "Golden store must have at least 50 samples"

    def test_golden_columns(self, feature_names):
        df = pd.read_csv(os.path.join(GOLDEN_DIR, "golden_samples.csv"))
        assert list(df.columns) == feature_names


@pytest.mark.unit
class TestConstraints:
    def test_constraints_file_exists(self):
        assert os.path.exists(os.path.join(CONFIG_DIR, "constraints.yaml"))

    def test_constraints_covers_all_features(self, feature_names):
        with open(os.path.join(CONFIG_DIR, "constraints.yaml")) as f:
            constraints = yaml.safe_load(f)
        missing = set(feature_names) - set(constraints.keys())
        assert not missing, f"Constraints missing for: {missing}"

    def test_constraints_min_lt_max(self, feature_names):
        with open(os.path.join(CONFIG_DIR, "constraints.yaml")) as f:
            constraints = yaml.safe_load(f)
        for feat in feature_names:
            assert constraints[feat]["min"] < constraints[feat]["max"], \
                f"{feat}: min must be < max"
