"""
experiment_tracker.py — Logs every training run to models/experiments.json.
Enables reproducibility and champion comparison in the Model Forge page.
"""
import json
import os
from datetime import datetime

import pandas as pd

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_FILE = os.path.join(BASE_DIR, "models", "experiments.json")


def _load_raw() -> list:
    if not os.path.exists(EXPERIMENTS_FILE):
        return []
    try:
        with open(EXPERIMENTS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def log(algo: str, cv_mean: float, cv_std: float,
        val_acc: float, val_f1: float, hyperparams: dict) -> None:
    """Append one training run to the experiments log."""
    runs = _load_raw()
    runs.append({
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algo":       algo,
        "cv_mean":    round(float(cv_mean),  4),
        "cv_std":     round(float(cv_std),   4),
        "val_acc":    round(float(val_acc),  4),
        "val_f1":     round(float(val_f1),   4),
        "hyperparams": hyperparams,
    })
    os.makedirs(os.path.dirname(EXPERIMENTS_FILE), exist_ok=True)
    with open(EXPERIMENTS_FILE, "w") as f:
        json.dump(runs, f, indent=2)


def load_df() -> pd.DataFrame:
    """Return all experiments as a DataFrame, newest first."""
    runs = _load_raw()
    if not runs:
        return pd.DataFrame(columns=[
            "timestamp", "algo", "cv_mean", "cv_std", "val_acc", "val_f1"
        ])
    df = pd.DataFrame(runs)
    df = df[["timestamp", "algo", "cv_mean", "cv_std", "val_acc", "val_f1"]]
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


def get_champion() -> dict:
    """Return the run with the highest validation accuracy."""
    runs = _load_raw()
    if not runs:
        return {}
    return max(runs, key=lambda r: r["val_acc"])
