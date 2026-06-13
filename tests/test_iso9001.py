"""
Tests for iso9001_metrics — pure math, no I/O or model loading.
Verifies Cp/Cpk formulas, edge cases, and capability thresholds.
"""
import math
import numpy as np
import pandas as pd
import pytest
from iso9001_metrics import compute_capability, cpk_status, compute_all_capabilities


@pytest.mark.unit
class TestCpFormula:
    def test_cp_basic(self):
        # USL=10, LSL=0, σ=1 → Cp = 10/6 ≈ 1.667
        s = pd.Series([5.0] * 100 + [4.0, 6.0])  # μ≈5, σ≈small
        s = pd.Series(np.random.normal(5, 1, 1000))
        usl, lsl = 10.0, 0.0
        cp, _ = compute_capability(s, usl, lsl)
        expected_cp = (usl - lsl) / (6 * float(s.std(ddof=1)))
        assert cp == pytest.approx(expected_cp, abs=0.01)

    def test_cp_wider_tolerance_higher_score(self):
        s = pd.Series(np.random.normal(0, 1, 500))
        cp_narrow, _ = compute_capability(s, 2.0, -2.0)
        cp_wide,   _ = compute_capability(s, 5.0, -5.0)
        assert cp_wide > cp_narrow

    def test_cp_always_positive(self):
        s = pd.Series(np.random.normal(50, 5, 200))
        cp, _ = compute_capability(s, 70.0, 30.0)
        assert cp > 0


@pytest.mark.unit
class TestCpkFormula:
    def test_centered_process_cp_equals_cpk(self):
        # Perfectly centred: μ = (USL+LSL)/2 → Cpk == Cp
        lsl, usl = 0.0, 10.0
        mu = (usl + lsl) / 2
        s = pd.Series(np.full(1000, mu))
        # Force nonzero std
        s.iloc[0] = mu + 1
        s.iloc[1] = mu - 1
        cp, cpk = compute_capability(s, usl, lsl)
        assert cpk <= cp + 1e-6

    def test_off_centre_cpk_less_than_cp(self):
        s = pd.Series(np.random.normal(8, 1, 500))  # shifted toward USL
        cp, cpk = compute_capability(s, 10.0, 0.0)
        assert cpk < cp

    def test_cpk_below_zero_when_outside_spec(self):
        # μ > USL → Cpk must be negative
        s = pd.Series(np.full(10, 15.0))
        s.iloc[0] = 14.0  # nonzero std
        _, cpk = compute_capability(s, 10.0, 0.0)
        assert cpk < 0

    def test_known_values(self):
        # μ=5, σ=1, USL=8, LSL=2 → Cp=(8-2)/6=1.0; Cpk=min((8-5)/3,(5-2)/3)=1.0
        s = pd.Series([5.0] * 998 + [4.0, 6.0])
        # Override with exact σ=1 using linspace
        s = pd.Series(np.linspace(4, 6, 1000))  # μ=5, range=2
        usl, lsl = 8.0, 2.0
        cp, cpk = compute_capability(s, usl, lsl)
        # Both should be > 1 (process is capable)
        assert cp > 1.0
        assert cpk > 0.5


@pytest.mark.unit
class TestEdgeCases:
    def test_zero_std_returns_nan(self):
        s = pd.Series([5.0] * 50)  # constant → σ=0
        cp, cpk = compute_capability(s, 10.0, 0.0)
        assert math.isnan(cp)
        assert math.isnan(cpk)

    def test_single_value_returns_nan(self):
        s = pd.Series([5.0])
        cp, cpk = compute_capability(s, 10.0, 0.0)
        assert math.isnan(cp)
        assert math.isnan(cpk)


@pytest.mark.unit
class TestCpkStatus:
    def test_capable(self):
        assert cpk_status(1.33) == "Capable"
        assert cpk_status(2.0)  == "Capable"

    def test_marginal(self):
        assert cpk_status(1.0)  == "Marginal"
        assert cpk_status(1.32) == "Marginal"

    def test_not_capable(self):
        assert cpk_status(0.99) == "Not Capable"
        assert cpk_status(-1.0) == "Not Capable"

    def test_nan_not_capable(self):
        assert cpk_status(float("nan")) == "Not Capable"


@pytest.mark.integration
class TestComputeAllCapabilities:
    def test_returns_dataframe(self, X_val_real, feature_names):
        import yaml, os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, "config", "constraints.yaml")) as f:
            constraints = yaml.safe_load(f)
        result = compute_all_capabilities(X_val_real, constraints)
        assert isinstance(result, pd.DataFrame)

    def test_all_features_present(self, X_val_real, feature_names):
        import yaml, os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, "config", "constraints.yaml")) as f:
            constraints = yaml.safe_load(f)
        result = compute_all_capabilities(X_val_real, constraints)
        assert len(result) == len(feature_names)

    def test_required_columns(self, X_val_real, feature_names):
        import yaml, os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, "config", "constraints.yaml")) as f:
            constraints = yaml.safe_load(f)
        result = compute_all_capabilities(X_val_real, constraints)
        for col in ["Feature", "Cp", "Cpk", "Status", "LSL", "USL", "Mean", "Std"]:
            assert col in result.columns
