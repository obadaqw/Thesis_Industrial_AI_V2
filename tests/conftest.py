"""
Shared fixtures — loaded once per test session to avoid re-training models.
"""
import os
import sys
import pytest
import joblib
import pandas as pd
import numpy as np

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, "models", "checkpoints")
PROC_DIR   = os.path.join(BASE, "data", "processed")


# ── Lightweight fixtures (no model loading) ────────────────────────────────

@pytest.fixture(scope="session")
def feature_names():
    return joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))


@pytest.fixture(scope="session")
def scaler():
    return joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))


@pytest.fixture(scope="session")
def model():
    return joblib.load(os.path.join(MODELS_DIR, "current_model.pkl"))


@pytest.fixture(scope="session")
def train_stats():
    return joblib.load(os.path.join(MODELS_DIR, "train_stats.pkl"))


@pytest.fixture(scope="session")
def X_train(feature_names):
    return pd.read_csv(os.path.join(PROC_DIR, "X_train.csv"))


@pytest.fixture(scope="session")
def y_train():
    raw = pd.read_csv(os.path.join(PROC_DIR, "y_train.csv")).values.ravel()
    return raw - 1 if raw.min() > 0 else raw


@pytest.fixture(scope="session")
def X_val(feature_names):
    return pd.read_csv(os.path.join(PROC_DIR, "X_val.csv"))


@pytest.fixture(scope="session")
def y_val():
    raw = pd.read_csv(os.path.join(PROC_DIR, "y_val.csv")).values.ravel()
    return raw - 1 if raw.min() > 0 else raw


@pytest.fixture(scope="session")
def X_val_real(scaler, X_val, feature_names):
    return pd.DataFrame(scaler.inverse_transform(X_val), columns=feature_names)


@pytest.fixture(scope="session")
def defect_row(X_val_real, y_val):
    """One defect sample in physical (unscaled) units — for CF RCA."""
    idx = int(np.where(~np.isin(y_val, [1, 2]))[0][0])
    return X_val_real.iloc[[idx]]


@pytest.fixture(scope="session")
def good_row(X_val_real, y_val):
    """One Target-class sample in physical (unscaled) units — for CF RCA."""
    idx = int(np.where(y_val == 2)[0][0])
    return X_val_real.iloc[[idx]]


@pytest.fixture(scope="session")
def defect_scaled_row(X_val, y_val):
    """One defect sample in MinMax-scaled space — for XAI engine."""
    idx = int(np.where(~np.isin(y_val, [1, 2]))[0][0])
    return X_val.iloc[[idx]]


@pytest.fixture(scope="session")
def good_scaled_row(X_val, y_val):
    """One Target-class sample in MinMax-scaled space — for XAI engine."""
    idx = int(np.where(y_val == 2)[0][0])
    return X_val.iloc[[idx]]


# ── Heavy fixtures (load full engines once) ────────────────────────────────

@pytest.fixture(scope="session")
def xai_engine():
    from xai_engine import XAIEngine
    return XAIEngine()


@pytest.fixture(scope="session")
def rca_surrogate():
    from rca_surrogate import RCASurrogate
    return RCASurrogate()


@pytest.fixture(scope="session")
def cf_rca():
    from counterfactual_rca import CounterfactualRCA
    return CounterfactualRCA()
