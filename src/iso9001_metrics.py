"""
iso9001_metrics.py — Pure process-capability functions (no I/O, no model deps).
Extracted here so they can be unit-tested independently of the Streamlit page.
"""
import numpy as np
import pandas as pd


def compute_capability(series: pd.Series, usl: float, lsl: float):
    """
    Returns (Cp, Cpk) for a data series given spec limits.

    Cp  = (USL - LSL) / (6σ)          — spread capability
    Cpk = min((USL-μ)/(3σ), (μ-LSL)/(3σ)) — centred capability

    Returns (nan, nan) when σ == 0 to avoid division by zero.
    """
    mu  = float(series.mean())
    sig = float(series.std(ddof=1))
    if sig == 0 or np.isnan(sig):
        return float("nan"), float("nan")
    cp  = (usl - lsl) / (6.0 * sig)
    cpk = min((usl - mu) / (3.0 * sig), (mu - lsl) / (3.0 * sig))
    return round(cp, 3), round(cpk, 3)


def cpk_status(cpk: float) -> str:
    if np.isnan(cpk) or cpk < 1.0:
        return "Not Capable"
    if cpk < 1.33:
        return "Marginal"
    return "Capable"


def compute_all_capabilities(X_real: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """Compute Cp/Cpk for every feature that appears in constraints."""
    rows = []
    for feat in X_real.columns:
        if feat not in constraints:
            continue
        lsl = constraints[feat]["min"]
        usl = constraints[feat]["max"]
        cp, cpk = compute_capability(X_real[feat], usl, lsl)
        rows.append({
            "Feature": feat,
            "LSL":     lsl,
            "USL":     usl,
            "Mean":    round(X_real[feat].mean(), 4),
            "Std":     round(X_real[feat].std(ddof=1), 4),
            "Cp":      cp,
            "Cpk":     cpk,
            "Status":  cpk_status(cpk),
        })
    return pd.DataFrame(rows)
