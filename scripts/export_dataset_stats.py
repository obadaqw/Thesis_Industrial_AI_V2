"""
export_dataset_stats.py — Dataset statistics export for thesis Chapter 3.

Reads exclusively from data/raw/raw_data.csv and data/processed/*.csv (already
produced by src/data_pipeline.py). Computes no new modelling result; every
number is derived at generation time from these source files so the thesis
text and the artifacts cannot drift apart.

Outputs (thesis_assets/data/):
  feature_specification.json   13 features: column, physical quantity, unit, stats
  class_distribution.json      class counts/pct, full + per split
  correlation_matrix.csv       Pearson correlation, 13x13, training split
  split_summary.json           n per split, proportions, seed, stratification check

Usage:
    python scripts/export_dataset_stats.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

RAW_PATH  = os.path.join(BASE, "data", "raw", "raw_data.csv")
PROC_DIR  = os.path.join(BASE, "data", "processed")
OUT_DIR   = os.path.join(BASE, "thesis_assets", "data")

CLASS_NAMES = {0: "Waste", 1: "Acceptable", 2: "Target", 3: "Inefficient"}
CONFORMING  = {1, 2}

# Physical quantity and unit derived from the column name and standard
# injection-molding process-parameter nomenclature (the ZDx/ZUx/SKx/SKs/Ms/
# Mm/APSs/APVs/CPn/SVo prefixes follow the German-machine-controller
# convention: Dosierzeit, Zykluszeit, Schliesskraft, Drehmoment, spez.
# Staudruck/Einspritzdruck, Umschaltpunkt, Schussvolumen). Only the
# temperature and time-family features have a unit confirmed with high
# confidence from the name and the value range alone; the paper (Polenta et
# al., 2022) does not tabulate units for the remaining channels, so those are
# left UNVERIFIED per project convention rather than guessed.
FEATURE_SPEC = {
    "Melt temperature": {
        "physical_quantity": "Polymer melt temperature",
        "unit": "degC",
    },
    "Mold temperature": {
        "physical_quantity": "Mold surface temperature",
        "unit": "degC",
    },
    "time_to_fill": {
        "physical_quantity": "Cavity filling time",
        "unit": "s",
    },
    "ZDx - Plasticizing time": {
        "physical_quantity": "Plasticizing (dosing) time",
        "unit": "s",
    },
    "ZUx - Cycle time": {
        "physical_quantity": "Total cycle time",
        "unit": "s",
    },
    "SKx - Closing force": {
        "physical_quantity": "Mold closing force",
        "unit": "UNVERIFIED",
    },
    "SKs - Clamping force peak value": {
        "physical_quantity": "Clamping force, peak value",
        "unit": "UNVERIFIED",
    },
    "Ms - Torque peak value current cycle": {
        "physical_quantity": "Screw drive torque, peak value",
        "unit": "UNVERIFIED",
    },
    "Mm - Torque mean value current cycle": {
        "physical_quantity": "Screw drive torque, mean value",
        "unit": "UNVERIFIED",
    },
    "APSs - Specific back pressure peak value": {
        "physical_quantity": "Specific back pressure, peak value",
        "unit": "UNVERIFIED",
    },
    "APVs - Specific injection pressure peak value": {
        "physical_quantity": "Specific injection pressure, peak value",
        "unit": "UNVERIFIED",
    },
    "CPn - Screw position at the end of hold pressure": {
        "physical_quantity": "Screw position at end of hold (cushion) pressure",
        "unit": "UNVERIFIED",
    },
    "SVo - Shot volume": {
        "physical_quantity": "Shot volume",
        "unit": "UNVERIFIED",
    },
}


def _load_raw():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Required source artifact missing: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def _load_split(name: str):
    x_path = os.path.join(PROC_DIR, f"X_{name}.csv")
    y_path = os.path.join(PROC_DIR, f"y_{name}.csv")
    for p in (x_path, y_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required source artifact missing: {p}")
    X = pd.read_csv(x_path)
    y_raw = pd.read_csv(y_path).values.ravel()
    y = y_raw - 1 if y_raw.min() > 0 else y_raw
    return X, y


def export_feature_specification(df_raw: pd.DataFrame, feature_names: list) -> dict:
    out = {}
    unverified = []
    for feat in feature_names:
        if feat not in FEATURE_SPEC:
            raise KeyError(
                f"No physical-quantity mapping registered for feature '{feat}'. "
                "Add an entry to FEATURE_SPEC before export."
            )
        series = df_raw[feat]
        spec = dict(FEATURE_SPEC[feat])
        spec.update({
            "column": feat,
            "min":    round(float(series.min()), 6),
            "max":    round(float(series.max()), 6),
            "mean":   round(float(series.mean()), 6),
            "median": round(float(series.median()), 6),
            "std":    round(float(series.std(ddof=1)), 6),
            "n":      int(series.count()),
        })
        if spec["unit"] == "UNVERIFIED":
            unverified.append(feat)
        out[feat] = spec

    result = {
        "features": out,
        "n_features": len(out),
        "units_unverified": unverified,
        "source": "data/raw/raw_data.csv (raw, unscaled)",
    }
    with open(os.path.join(OUT_DIR, "feature_specification.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def export_class_distribution(df_raw: pd.DataFrame, splits: dict) -> dict:
    def _dist(y_series):
        y_series = pd.Series(y_series)
        counts = y_series.value_counts().sort_index()
        total = int(counts.sum())
        dist = {}
        for cls_idx in range(4):
            n = int(counts.get(cls_idx, 0))
            dist[CLASS_NAMES[cls_idx]] = {
                "count": n,
                "pct": round(100.0 * n / total, 2) if total else 0.0,
            }
        conforming_n = sum(dist[CLASS_NAMES[c]]["count"] for c in CONFORMING)
        return {
            "n_total": total,
            "by_class": dist,
            "conforming": {
                "count": conforming_n,
                "pct": round(100.0 * conforming_n / total, 2) if total else 0.0,
                "definition": "{Acceptable, Target}",
            },
            "non_conforming": {
                "count": total - conforming_n,
                "pct": round(100.0 * (total - conforming_n) / total, 2) if total else 0.0,
                "definition": "{Waste, Inefficient}",
            },
        }

    # Full dataset: raw 'quality' column is 1-based; convert to 0-based.
    y_full_raw = df_raw["quality"].values
    y_full = y_full_raw - 1 if y_full_raw.min() > 0 else y_full_raw

    result = {
        "full_dataset": _dist(y_full),
        "splits": {name: _dist(y) for name, (_, y) in splits.items()},
    }
    with open(os.path.join(OUT_DIR, "class_distribution.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def export_correlation_matrix(X_train_real: pd.DataFrame) -> pd.DataFrame:
    corr = X_train_real.corr(method="pearson")
    corr.to_csv(os.path.join(OUT_DIR, "correlation_matrix.csv"))
    return corr


def export_split_summary(df_raw: pd.DataFrame, splits: dict) -> dict:
    n_total = len(df_raw)
    sizes = {name: len(y) for name, (_, y) in splits.items()}

    # Stratification confirmation: compare per-class proportions across
    # splits against the full-dataset proportions (data_pipeline.py uses
    # sklearn's stratify= argument on the raw 1-based quality column).
    y_full_raw = df_raw["quality"].values
    y_full = y_full_raw - 1 if y_full_raw.min() > 0 else y_full_raw
    full_props = pd.Series(y_full).value_counts(normalize=True).sort_index()

    max_abs_dev = 0.0
    for name, (_, y) in splits.items():
        split_props = pd.Series(y).value_counts(normalize=True).sort_index()
        dev = (split_props - full_props).abs().max()
        max_abs_dev = max(max_abs_dev, float(dev))

    result = {
        "n_total": n_total,
        "sizes": sizes,
        "proportions": {name: round(n / n_total, 4) for name, n in sizes.items()},
        "nominal_split": "60% train / 20% val / 20% test",
        "random_state": 42,
        "stratified": True,
        "stratify_column": "quality",
        "max_abs_class_proportion_deviation": round(max_abs_dev, 4),
        "stratification_confirmed": bool(max_abs_dev < 0.02),
        "source_script": "src/data_pipeline.py",
    }
    with open(os.path.join(OUT_DIR, "split_summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Dataset statistics export ===")

    df_raw = _load_raw()
    feature_names = [c for c in df_raw.columns if c != "quality"]

    splits = {}
    real_splits = {}
    for name in ("train", "val", "test"):
        X_scaled, y = _load_split(name)
        splits[name] = (X_scaled, y)
        real_splits[name] = X_scaled  # inverse-transform done below for train

    # Inverse-transform X_train to real units for feature spec + correlation.
    import joblib
    scaler = joblib.load(os.path.join(BASE, "models", "checkpoints", "scaler.pkl"))
    X_train_real = pd.DataFrame(
        scaler.inverse_transform(splits["train"][0]), columns=feature_names
    )

    print("Exporting feature_specification.json ...")
    spec = export_feature_specification(df_raw, feature_names)
    print(f"  {len(spec['units_unverified'])}/13 units UNVERIFIED: "
          f"{spec['units_unverified']}")

    print("Exporting class_distribution.json ...")
    export_class_distribution(df_raw, splits)

    print("Exporting correlation_matrix.csv (training split) ...")
    export_correlation_matrix(X_train_real)

    print("Exporting split_summary.json ...")
    summary = export_split_summary(df_raw, splits)
    print(f"  sizes={summary['sizes']}  "
          f"stratification_confirmed={summary['stratification_confirmed']}")

    print(f"\nDone. Files written to {OUT_DIR}")


if __name__ == "__main__":
    main()
