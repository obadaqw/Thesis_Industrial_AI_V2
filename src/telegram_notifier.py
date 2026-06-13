"""
telegram_notifier.py — Push notifications via Telegram Bot API.

Setup:
  1. Message @BotFather on Telegram → /newbot → copy token
  2. Start a chat with your bot (or add it to a group)
  3. Get your chat_id: https://api.telegram.org/bot<TOKEN>/getUpdates
  4. Set in .env:
       TELEGRAM_BOT_TOKEN=<token>
       TELEGRAM_CHAT_ID=<chat_id>

All methods return True on success, False on failure/not-configured.
They never raise exceptions — safe to call from any page.
"""

import os
import threading
import requests
from datetime import datetime
from dotenv import load_dotenv
from config import get_secret

load_dotenv()

_API = "https://api.telegram.org/bot{token}/{method}"
_MAX_LEN = 4096


class TelegramNotifier:

    def __init__(self):
        self.token   = get_secret("TELEGRAM_BOT_TOKEN")
        self.chat_id = get_secret("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)
        status = "ready ✅" if self.enabled else "disabled (token/chat_id not set)"
        print(f"📱 TelegramNotifier {status}")

    # ── Core send ──────────────────────────────────────────────────────────

    def _send_sync(self, text: str, parse_mode: str = "HTML") -> bool:
        """Blocking HTTP send — used internally and for send_test_message."""
        if not self.enabled:
            return False
        try:
            r = requests.post(
                _API.format(token=self.token, method="sendMessage"),
                json={
                    "chat_id":    self.chat_id,
                    "text":       text[:_MAX_LEN],
                    "parse_mode": parse_mode,
                },
                timeout=5,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Fire-and-forget: launches a daemon thread, returns True immediately."""
        if not self.enabled:
            return False
        threading.Thread(
            target=self._send_sync, args=(text, parse_mode), daemon=True
        ).start()
        return True

    # ── Notification types ─────────────────────────────────────────────────

    def send_defect_alert(self, cycle_id: int, quality_class: int,
                          confidence: float) -> bool:
        labels = {0: "Waste ❌", 1: "Acceptable ⚠️", 2: "Target ✅", 3: "Inefficient 🔶"}
        ts = datetime.now().strftime("%H:%M:%S")
        text = (
            f"🚨 <b>DEFECT DETECTED</b>  [{ts}]\n"
            f"{'─'*28}\n"
            f"Cycle : <code>{cycle_id}</code>\n"
            f"Quality : <b>{labels.get(quality_class, str(quality_class))}</b>\n"
            f"Confidence : <b>{confidence:.1%}</b>\n"
            f"{'─'*28}\n"
            f"⚡ Open RCA Investigator for root-cause analysis."
        )
        return self._send(text)

    def send_rca_alert(self, cycle_id: int, tier: int, adjustments: list,
                       validator_ok: bool, cf_confidence: float) -> bool:
        tier_label = {
            1: "🔵 Tier 1 — SHAP+LIME",
            2: "🟡 Tier 2 — NN-Anchored",
            3: "🔴 Tier 3 — ESCALATION",
        }.get(tier, f"Tier {tier}")
        ts  = datetime.now().strftime("%H:%M:%S")
        vok = "✅ Confirmed" if validator_ok else "⚠️ Unconfirmed"
        adj_lines = "\n".join(
            f"  {a['direction']} <b>{a['feature']}</b>: "
            f"{a['current']} → {a['suggested']} (Δ={a['delta']})"
            for a in adjustments[:5]
        ) or "  —"
        text = (
            f"🔬 <b>RCA COMPLETED</b>  [{ts}]\n"
            f"{'─'*28}\n"
            f"Cycle      : <code>{cycle_id}</code>\n"
            f"Tier       : <b>{tier_label}</b>\n"
            f"CF Conf    : <b>{cf_confidence:.1%}</b>\n"
            f"Validator  : <b>{vok}</b>\n"
            f"{'─'*28}\n"
            f"<b>Adjustments:</b>\n{adj_lines}"
        )
        if tier == 3:
            text += "\n\n🚨 <b>Escalate to maintenance — manual inspection required.</b>"
        return self._send(text)

    def send_oee_alert(self, oee: float, threshold: float,
                       total_cycles: int, good_cycles: int) -> bool:
        ts = datetime.now().strftime("%H:%M:%S")
        scrap_rate = (total_cycles - good_cycles) / max(1, total_cycles)
        text = (
            f"⚠️ <b>OEE BELOW THRESHOLD</b>  [{ts}]\n"
            f"{'─'*28}\n"
            f"OEE       : <b>{oee:.1%}</b>  (limit: {threshold:.0%})\n"
            f"Scrap Rate: <b>{scrap_rate:.1%}</b>\n"
            f"Good/Total: {good_cycles}/{total_cycles}\n"
            f"{'─'*28}\n"
            f"⚡ Review Digital Twin dashboard."
        )
        return self._send(text)

    def send_shift_report(self, report_text: str, cycle_id: int = None) -> bool:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        header = f"📄 <b>SHIFT REPORT</b>  [{ts}]"
        if cycle_id is not None:
            header += f"  — Cycle <code>{cycle_id}</code>"
        full = f"{header}\n{'─'*28}\n{report_text}"
        return self._send(full)

    def send_iso_alert(self, feature: str, cpk: float,
                       threshold: float = 1.0) -> bool:
        ts = datetime.now().strftime("%H:%M:%S")
        text = (
            f"📊 <b>ISO 9001 PROCESS ALERT</b>  [{ts}]\n"
            f"{'─'*28}\n"
            f"Feature : <b>{feature}</b>\n"
            f"Cpk     : <b>{cpk:.3f}</b>  (min: {threshold})\n"
            f"Status  : ❌ <b>Process NOT capable</b>\n"
            f"{'─'*28}\n"
            f"⚡ Inspect and recalibrate sensor."
        )
        return self._send(text)

    def send_test_message(self) -> bool:
        """Blocking send so the UI gets immediate success/failure feedback."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self._send_sync(
            f"✅ <b>Thesis Industrial AI V2</b>\n"
            f"Telegram notifications connected.\n"
            f"<i>{ts}</i>"
        )
