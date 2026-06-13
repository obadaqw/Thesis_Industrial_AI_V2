"""
Tests for DigitalTwin — OEE math, good/bad cycle accounting, reset.
Pure unit tests: no model or data loading required.
"""
import pytest
from digital_twin import DigitalTwin


@pytest.fixture
def twin():
    return DigitalTwin()


@pytest.mark.unit
class TestGoodBadCycles:
    def test_class2_increments_good(self, twin):
        twin.update_metrics(2)
        assert twin.state["good_cycles"] == 1
        assert twin.state["bad_cycles"]  == 0

    def test_class0_increments_bad(self, twin):
        twin.update_metrics(0)
        assert twin.state["good_cycles"] == 0
        assert twin.state["bad_cycles"]  == 1

    def test_class1_increments_bad(self, twin):
        # Acceptable (class 1) is NOT Target — counts as bad for OEE quality
        twin.update_metrics(1)
        assert twin.state["bad_cycles"] == 1

    def test_class3_increments_bad(self, twin):
        twin.update_metrics(3)
        assert twin.state["bad_cycles"] == 1


@pytest.mark.unit
class TestOEEMath:
    def test_quality_ratio_7_good_3_bad(self, twin):
        for _ in range(7): twin.update_metrics(2)
        for _ in range(3): twin.update_metrics(0)
        assert twin.state["quality"] == pytest.approx(0.7, abs=1e-9)

    def test_oee_equals_a_times_p_times_q(self, twin):
        state = twin.update_metrics(2)
        expected = state["availability"] * state["performance"] * state["quality"]
        assert state["oee"] == pytest.approx(expected, abs=1e-9)

    def test_oee_in_unit_interval(self, twin):
        for pred in [2, 0, 2, 3, 2]:
            state = twin.update_metrics(pred)
        assert 0.0 <= state["oee"] <= 1.0

    def test_perfect_quality_100_good(self, twin):
        for _ in range(20): twin.update_metrics(2)
        assert twin.state["quality"] == pytest.approx(1.0, abs=1e-9)

    def test_total_cycles_increments(self, twin):
        for i in range(5):
            twin.update_metrics(2)
        assert twin.state["total_cycles"] == 5

    def test_availability_degrades(self, twin):
        initial = twin.state["availability"]
        for _ in range(50):
            twin.update_metrics(2)
        assert twin.state["availability"] < initial


@pytest.mark.unit
class TestReset:
    def test_reset_clears_total_cycles(self, twin):
        twin.update_metrics(2)
        twin.reset_machine()
        assert twin.state["total_cycles"] == 0

    def test_reset_clears_good_cycles(self, twin):
        twin.update_metrics(2)
        twin.reset_machine()
        assert twin.state["good_cycles"] == 0

    def test_reset_clears_bad_cycles(self, twin):
        twin.update_metrics(0)
        twin.reset_machine()
        assert twin.state["bad_cycles"] == 0

    def test_reset_restores_availability(self, twin):
        for _ in range(100): twin.update_metrics(2)
        twin.reset_machine()
        assert twin.state["availability"] == pytest.approx(0.98)

    def test_reset_restores_quality(self, twin):
        twin.update_metrics(0)
        twin.reset_machine()
        assert twin.state["quality"] == pytest.approx(1.0)
