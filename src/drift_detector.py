"""
drift_detector.py — Statistical drift detection (Decision #17).

Two methods:
  1. PSI (Population Stability Index) — feature distribution drift
     < 0.10 : No drift
     0.10-0.20 : Moderate drift — monitor
     > 0.20 : Significant drift — retrain

  2. Page-Hinkley Test — sequential concept drift on model confidence stream
     Raises alarm when cumulative deviation exceeds threshold.
"""
import numpy as np
import pandas as pd

PSI_BINS = 10
PSI_MODERATE  = 0.10
PSI_CRITICAL  = 0.20

PH_DELTA      = 0.005   # allowed mean shift before accumulation starts
PH_THRESHOLD  = 50      # cumulative sum threshold to flag drift
PH_ALPHA      = 0.9999  # EMA smoothing factor for running mean


def _psi_single(expected: np.ndarray, actual: np.ndarray, bins: int = PSI_BINS) -> float:
    """PSI between two 1-D arrays using percentile-based binning."""
    breaks = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breaks[0]  = -np.inf
    breaks[-1] =  np.inf

    exp_cnt = np.histogram(expected, bins=breaks)[0]
    act_cnt = np.histogram(actual,   bins=breaks)[0]

    exp_pct = np.where(exp_cnt == 0, 1e-4, exp_cnt / len(expected))
    act_pct = np.where(act_cnt == 0, 1e-4, act_cnt / len(actual))

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 4)


def psi_status(psi: float) -> str:
    if psi < PSI_MODERATE:
        return "stable"
    if psi < PSI_CRITICAL:
        return "moderate"
    return "critical"


def psi_emoji(psi: float) -> str:
    s = psi_status(psi)
    return {"stable": "✅ Stable", "moderate": "⚠️ Monitor", "critical": "🚨 Retrain"}[s]


class DriftDetector:

    def __init__(self, X_train: pd.DataFrame, feature_names: list):
        self.reference     = X_train.values
        self.feature_names = feature_names

    def compute_all_psi(self, X_current: pd.DataFrame) -> pd.DataFrame:
        """Compute PSI for every feature between reference and current batch."""
        rows = []
        for i, feat in enumerate(self.feature_names):
            psi = _psi_single(self.reference[:, i], X_current.values[:, i])
            rows.append({
                "Feature": feat,
                "PSI":     psi,
                "Status":  psi_emoji(psi),
            })
        return pd.DataFrame(rows).sort_values("PSI", ascending=False)

    def feature_psi(self, feature: str, X_current: pd.DataFrame) -> float:
        i = self.feature_names.index(feature)
        return _psi_single(self.reference[:, i], X_current.values[:, i])

    @staticmethod
    def page_hinkley(confidence_stream: np.ndarray,
                     delta: float = PH_DELTA,
                     threshold: float = PH_THRESHOLD,
                     alpha: float = PH_ALPHA) -> list:
        """
        Page-Hinkley test on a model confidence stream.
        Returns indices where drift was detected.
        """
        if len(confidence_stream) == 0:
            return []

        m   = float(confidence_stream[0])
        M   = 0.0
        pts = []

        for i, x in enumerate(confidence_stream):
            m = alpha * m + (1 - alpha) * float(x)
            M = max(0.0, M + float(x) - m - delta)
            if M > threshold:
                pts.append(i)
                M = 0.0   # reset after detection

        return pts

    def overall_status(self, psi_df: pd.DataFrame) -> str:
        """Overall system drift status from a PSI DataFrame."""
        if (psi_df["PSI"] >= PSI_CRITICAL).any():
            return "critical"
        if (psi_df["PSI"] >= PSI_MODERATE).any():
            return "moderate"
        return "stable"
