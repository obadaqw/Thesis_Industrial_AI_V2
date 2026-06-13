"""
Tests for TelegramNotifier — verifies graceful degradation when unconfigured
and API contract (all methods return bool, never raise).
No real network calls are made.
"""
import os
import pytest
from unittest.mock import patch
from telegram_notifier import TelegramNotifier


@pytest.fixture
def disabled_notifier():
    """Notifier with no credentials — simulates missing .env."""
    with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}, clear=False):
        return TelegramNotifier()


@pytest.fixture
def fake_notifier():
    """Notifier with fake credentials — enabled=True but network will fail."""
    with patch.dict(os.environ,
                    {"TELEGRAM_BOT_TOKEN": "fake:token", "TELEGRAM_CHAT_ID": "999"},
                    clear=False):
        n = TelegramNotifier()
    return n


@pytest.mark.unit
class TestDisabledBehavior:
    def test_disabled_when_no_token(self, disabled_notifier):
        assert disabled_notifier.enabled is False

    def test_send_defect_returns_false(self, disabled_notifier):
        result = disabled_notifier.send_defect_alert(0, 0, 0.9)
        assert result is False

    def test_send_rca_returns_false(self, disabled_notifier):
        result = disabled_notifier.send_rca_alert(0, 1, [], False, 0.6)
        assert result is False

    def test_send_oee_returns_false(self, disabled_notifier):
        result = disabled_notifier.send_oee_alert(0.5, 0.7, 100, 50)
        assert result is False

    def test_send_shift_report_returns_false(self, disabled_notifier):
        result = disabled_notifier.send_shift_report("report text", cycle_id=1)
        assert result is False

    def test_send_iso_returns_false(self, disabled_notifier):
        result = disabled_notifier.send_iso_alert("Melt temperature", 0.8)
        assert result is False

    def test_send_test_message_returns_false(self, disabled_notifier):
        result = disabled_notifier.send_test_message()
        assert result is False

    def test_no_exception_on_any_method(self, disabled_notifier):
        """All methods must be safe to call regardless of configuration."""
        disabled_notifier.send_defect_alert(0, 0, 0.9)
        disabled_notifier.send_rca_alert(0, 3, [], False, 0.0)
        disabled_notifier.send_oee_alert(0.4, 0.7, 200, 80)
        disabled_notifier.send_shift_report("x")
        disabled_notifier.send_iso_alert("feature", 0.5, 1.0)
        disabled_notifier.send_test_message()


@pytest.mark.unit
class TestAPIContract:
    def test_all_methods_exist(self, disabled_notifier):
        required = [
            "send_defect_alert", "send_rca_alert", "send_oee_alert",
            "send_shift_report", "send_iso_alert", "send_test_message",
        ]
        for method in required:
            assert hasattr(disabled_notifier, method), f"Missing method: {method}"

    def test_enabled_flag_is_bool(self, disabled_notifier):
        assert isinstance(disabled_notifier.enabled, bool)

    def test_fake_credentials_set_enabled_true(self, fake_notifier):
        assert fake_notifier.enabled is True

    def test_send_with_fake_creds_returns_false_gracefully(self, fake_notifier):
        # Network will fail → _send catches exception and returns False
        result = fake_notifier.send_test_message()
        assert result is False

    def test_adjustments_with_content(self, disabled_notifier):
        adjustments = [
            {"direction": "↑", "feature": "Melt temperature",
             "current": 105.0, "suggested": 110.0, "delta": 5.0}
        ]
        result = disabled_notifier.send_rca_alert(1, 1, adjustments, True, 0.75)
        assert result is False
