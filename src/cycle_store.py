"""
cycle_store.py — SQLite cycle history for ISO 9001 traceability (Decision #15).
Every RCA run is logged with timestamp, prediction, CF result, and health score.
"""
import os
import sqlite3
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "cycle_history.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    cycle_id     INTEGER,
    prediction   INTEGER,
    confidence   REAL,
    rca_tier     INTEGER,
    rca_status   TEXT,
    cf_confidence REAL,
    validator_ok  INTEGER
);
"""


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def log_cycle(cycle_id: int, prediction: int, confidence: float,
              rca_tier: int, rca_status: str,
              cf_confidence: float = 0.0, validator_ok: bool = False) -> None:
    conn = _connect()
    conn.execute(
        """INSERT INTO cycles
           (timestamp, cycle_id, prediction, confidence,
            rca_tier, rca_status, cf_confidence, validator_ok)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            int(cycle_id), int(prediction), float(confidence),
            int(rca_tier), str(rca_status),
            float(cf_confidence), int(validator_ok),
        ),
    )
    conn.commit()
    conn.close()


def get_recent(n: int = 100) -> pd.DataFrame:
    """Return the n most recent cycles as a DataFrame."""
    conn = _connect()
    df = pd.read_sql_query(
        "SELECT * FROM cycles ORDER BY id DESC LIMIT ?", conn, params=(n,)
    )
    conn.close()
    return df


def get_stats() -> dict:
    """Aggregate statistics across all logged cycles."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cycles")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM cycles WHERE prediction IN (1,2)")
    good = cur.fetchone()[0]

    cur.execute(
        "SELECT rca_tier, COUNT(*) FROM cycles GROUP BY rca_tier ORDER BY rca_tier"
    )
    tier_counts = dict(cur.fetchall())
    conn.close()

    return {
        "total":       total,
        "good":        good,
        "fpy":         round(good / max(1, total), 4),
        "tier_counts": tier_counts,
    }


def clear_history() -> None:
    conn = _connect()
    conn.execute("DELETE FROM cycles")
    conn.commit()
    conn.close()
