"""
config.py — Unified secret/config resolution.

Priority order:
  1. st.secrets (Streamlit Cloud / secrets.toml)
  2. os.environ  (local .env via python-dotenv)
  3. default     (empty string)

Usage:
    from config import get_secret
    key = get_secret("GROQ_API_KEY")
"""
import os


def get_secret(key: str, default: str = "") -> str:
    """Return config value, checking st.secrets before env vars."""
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return str(val).strip()
    except Exception:
        pass
    return os.environ.get(key, default).strip()
