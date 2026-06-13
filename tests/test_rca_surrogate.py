"""
Tests for RCASurrogate — component map coverage, z-score anomaly detection,
health score clamping, and result schema.
"""
import pytest
import pandas as pd
import numpy as np


EXPECTED_FEATURES = [
    "Melt temperature", "Mold temperature", "time_to_fill",
    "ZDx - Plasticizing time", "ZUx - Cycle time", "SKx - Closing force",
    "SKs - Clamping force peak value", "Ms - Torque peak value current cycle",
    "Mm - Torque mean value current cycle",
    "APSs - Specific back pressure peak value",
    "APVs - Specific injection pressure peak value",
    "CPn - Screw position at the end of hold pressure", "SVo - Shot volume",
]

REQUIRED_RESULT_KEYS = {"critical_anomalies", "warnings", "system_health_score"}


@pytest.mark.unit
class TestComponentMap:
    def test_all_13_features_mapped(self, rca_surrogate):
        mapped = set(rca_surrogate.COMPONENT_MAP.keys())
        missing = set(EXPECTED_FEATURES) - mapped
        assert not missing, f"COMPONENT_MAP missing features: {missing}"

    def test_no_extra_unmapped_features(self, rca_surrogate, feature_names):
        for feat in feature_names:
            assert feat in rca_surrogate.COMPONENT_MAP, \
                f"Feature '{feat}' not in COMPONENT_MAP"

    def test_components_are_strings(self, rca_surrogate):
        for feat, comp in rca_surrogate.COMPONENT_MAP.items():
            assert isinstance(comp, str) and len(comp) > 0


@pytest.mark.integration
class TestAnalyzeCycle:
    def _mean_row(self, rca_surrogate, feature_names, train_stats):
        vals = train_stats["mean"].values
        return pd.DataFrame([vals], columns=feature_names)

    def _fault_row(self, rca_surrogate, feature_names, train_stats, sigma=5):
        vals = train_stats["mean"].values.copy()
        vals[0] += sigma * train_stats["std"].values[0]
        return pd.DataFrame([vals], columns=feature_names)

    def test_required_keys(self, rca_surrogate, feature_names, train_stats):
        row = self._mean_row(rca_surrogate, feature_names, train_stats)
        result = rca_surrogate.analyze_cycle(row)
        assert REQUIRED_RESULT_KEYS.issubset(result.keys())

    def test_normal_cycle_full_health(self, rca_surrogate, feature_names, train_stats):
        row = self._mean_row(rca_surrogate, feature_names, train_stats)
        result = rca_surrogate.analyze_cycle(row)
        assert result["system_health_score"] == pytest.approx(100.0)
        assert len(result["critical_anomalies"]) == 0

    def test_fault_cycle_has_critical_anomaly(self, rca_surrogate, feature_names, train_stats):
        row = self._fault_row(rca_surrogate, feature_names, train_stats, sigma=5)
        result = rca_surrogate.analyze_cycle(row)
        assert len(result["critical_anomalies"]) >= 1

    def test_fault_cycle_health_below_100(self, rca_surrogate, feature_names, train_stats):
        row = self._fault_row(rca_surrogate, feature_names, train_stats, sigma=5)
        result = rca_surrogate.analyze_cycle(row)
        assert result["system_health_score"] < 100.0

    def test_health_clamped_to_zero(self, rca_surrogate, feature_names, train_stats):
        # Inject faults on all features → health cannot go below 0
        vals = train_stats["mean"].values.copy()
        vals += 10 * train_stats["std"].values  # +10σ everywhere
        row = pd.DataFrame([vals], columns=feature_names)
        result = rca_surrogate.analyze_cycle(row)
        assert result["system_health_score"] >= 0.0

    def test_health_never_above_100(self, rca_surrogate, feature_names, train_stats):
        row = self._mean_row(rca_surrogate, feature_names, train_stats)
        result = rca_surrogate.analyze_cycle(row)
        assert result["system_health_score"] <= 100.0

    def test_critical_anomaly_schema(self, rca_surrogate, feature_names, train_stats):
        row = self._fault_row(rca_surrogate, feature_names, train_stats, sigma=5)
        result = rca_surrogate.analyze_cycle(row)
        if result["critical_anomalies"]:
            a = result["critical_anomalies"][0]
            for key in ("sensor", "component", "value", "deviation", "action"):
                assert key in a

    def test_missing_feature_returns_error(self, rca_surrogate):
        bad_row = pd.DataFrame([{"NonExistent": 1.0}])
        result = rca_surrogate.analyze_cycle(bad_row)
        assert result.get("status") == "Error"
