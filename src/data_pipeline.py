"""
Module: data_pipeline.py
Responsibility: Logic implementation for data_pipeline.
Reference: Thesis Architecture Document.
"""
import pandas as pd
import numpy as np
import os
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "raw_data.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "checkpoints")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_clean_data():
    """
    Loads raw data and performs initial cleaning.
    """
    print(f"🔄 Loading raw data from: {RAW_DATA_PATH}")

    # 1. Universal Loader (Handles encoding issues)
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except UnicodeDecodeError:
        df = pd.read_csv(RAW_DATA_PATH, encoding='cp1252')

    # 2. Clean Column Names
    # Remove leading/trailing spaces which cause "KeyError" bugs
    df.columns = [c.strip() for c in df.columns]
    print(f"✅ Loaded {len(df)} rows. Columns cleaned.")

    return df


def run_pipeline():
    print("🚀 Starting Enterprise Data Pipeline...")

    # 1. Load
    df = load_and_clean_data()

    # 2. Separate Features (X) and Target (y)
    # NOTE: Adjust 'quality' if your target column has a different name
    target_col = 'quality'

    # Drop ID columns that confuse the AI
    drop_cols = ['id', 'cycle_id', 'row_number', 'quality_type', 'defect_type']
    drop_cols = [c for c in drop_cols if c in df.columns]  # Only drop if they exist

    if target_col not in df.columns:
        raise ValueError(f"❌ CRITICAL: Target column '{target_col}' not found in CSV.")

    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]

    # 3. Stratified Split (60% Train, 20% Val, 20% Test)
    # First, split into Train (60%) and Temp (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    # Then split Temp into Val (50% of temp = 20% total) and Test (50% of temp = 20% total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"📊 Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 4. Scaling — MinMaxScaler to (-1, 1) as per thesis spec.
    # Fit ONLY on training data to prevent leakage.
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # Save raw training statistics for z-score anomaly detection in RCA
    # (independent of scaler type, so RCA z-scores remain meaningful)
    train_stats = {
        'mean': X_train.mean(),
        'std': X_train.std()
    }
    joblib.dump(train_stats, os.path.join(MODELS_DIR, "train_stats.pkl"))

    # 5. Save Artifacts
    # We save the datasets as CSVs for debugging transparency
    X_train_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)

    X_val_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_val.csv"), index=False)
    y_val.to_csv(os.path.join(PROCESSED_DIR, "y_val.csv"), index=False)

    X_test_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    # CRITICAL: Save the scaler logic so we can "Un-Scale" later in the UI
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # Save the feature names list for the SHAP module
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_names.pkl"))

    print("✅ Pipeline Complete. Artifacts saved to /data/processed and /models/checkpoints")


if __name__ == "__main__":
    run_pipeline()