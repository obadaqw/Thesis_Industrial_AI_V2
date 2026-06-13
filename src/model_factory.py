"""
Module: model_factory.py
Responsibility: Logic implementation for model_factory.
Reference: Thesis Architecture Document.
"""

import pandas as pd
import joblib
import os
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import experiment_tracker

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "checkpoints")


# THE MODEL ZOO (6 Candidates as per Thesis Benchmark)
def get_model_architecture(algo_name):
    """Returns the un-trained model instance based on user selection."""
    algo_name = algo_name.upper()

    if algo_name == "RF":
        # Thesis-tuned hyperparameters (selected via stratified 5-fold CV)
        return RandomForestClassifier(
            n_estimators=151, max_depth=79, criterion='entropy',
            min_samples_leaf=2, min_samples_split=4,
            n_jobs=-1, random_state=42
        )
    elif algo_name == "XGB":
        return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    elif algo_name == "SVM":
        return SVC(kernel='rbf', probability=True, random_state=42)  # Prob=True is needed for SHAP
    elif algo_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif algo_name == "GB":
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    elif algo_name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    elif algo_name == "DT":
        return DecisionTreeClassifier(max_depth=10, random_state=42)
    else:
        raise ValueError(f"❌ Unknown Algorithm: {algo_name}")


def load_processed_data():
    print("🔄 Loading processed data...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
    X_val = pd.read_csv(os.path.join(PROCESSED_DIR, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(PROCESSED_DIR, "y_val.csv")).values.ravel()

    # XGBoost requires class labels to be integers starting from 0
    # Our data might be 1-4. We fix this locally for training if needed.
    if y_train.min() > 0:
        y_train = y_train - 1
        y_val = y_val - 1

    return X_train, y_train, X_val, y_val


def train_model(algo_name="RF"):  # RF is the thesis-selected champion
    print(f"🚀 Initializing Model Factory for: {algo_name}...")

    # 1. Load Data
    X_train, y_train, X_val, y_val = load_processed_data()

    # 2. Get Architecture
    model = get_model_architecture(algo_name)

    # 3. Train
    print(f"⚙️  Training {algo_name}...")
    model.fit(X_train, y_train)

    # 4. Stratified 5-Fold CV on training data (thesis benchmark requirement)
    print("🔄 Running Stratified 5-Fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"   CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  (folds: {np.round(cv_scores, 4)})")

    # 5. Validate on held-out validation set
    print("🧪 Validating performance on held-out set...")
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')

    print("-" * 40)
    print(f"✅ Training Complete: {algo_name}")
    print(f"📊 CV Accuracy:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"📊 Val Accuracy: {acc:.4f}")
    print(f"⚖️  Weighted F1:  {f1:.4f}")
    print("-" * 40)

    # 5. Save (We overwrite 'current_model.pkl' so the UI always picks the latest one)
    save_path = os.path.join(MODELS_DIR, "current_model.pkl")
    joblib.dump(model, save_path)

    # Also save with specific name for benchmarking history
    history_path = os.path.join(MODELS_DIR, f"{algo_name}_model.pkl")
    joblib.dump(model, history_path)

    print(f"💾 Model saved as 'current_model.pkl' (Active) and '{algo_name}_model.pkl' (Archive)")

    # Log experiment for comparison dashboard
    try:
        hyperparams = {k: v for k, v in model.get_params().items()
                       if not callable(v)}
        experiment_tracker.log(algo_name, cv_scores.mean(), cv_scores.std(),
                               acc, f1, hyperparams)
        print("📋 Experiment logged to models/experiments.json")
    except Exception as ex:
        print(f"   ⚠️ Experiment logging failed (non-fatal): {ex}")


if __name__ == "__main__":
    # Allow running from command line: python src/model_factory.py --algo XGB
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="RF", help="Choose: RF (champion), XGB, SVM, KNN, GB, MLP, DT")
    args = parser.parse_args()

    train_model(args.algo)