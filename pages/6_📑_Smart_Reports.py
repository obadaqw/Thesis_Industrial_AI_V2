import streamlit as st
import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from llm_wrapper import LLMWrapper
from rca_surrogate import RCASurrogate
from counterfactual_rca import CounterfactualRCA
from telegram_notifier import TelegramNotifier

st.set_page_config(page_title="Smart Reports", page_icon="📝", layout="wide")

st.title("📝 Generative AI Shift Reports")
st.markdown("### Automated Technical Reporting — Llama 3 on Groq + Counterfactual Context")


@st.cache_resource(show_spinner="Loading engines…")
def load_engines():
    scaler        = joblib.load(os.path.join("models", "checkpoints", "scaler.pkl"))
    feature_names = joblib.load(os.path.join("models", "checkpoints", "feature_names.pkl"))
    return LLMWrapper(), RCASurrogate(), CounterfactualRCA(), TelegramNotifier(), scaler, feature_names


@st.cache_data
def load_val_data():
    return pd.read_csv(os.path.join("data", "processed", "X_val.csv"))


try:
    llm_engine, rca_engine, cf_engine, notifier, scaler, feature_names = load_engines()
    X_val_scaled = load_val_data()
except Exception as e:
    st.error(f"Engine init failed: {e}")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 1. Select Cycle & Context")

    sample_id  = st.number_input("Cycle ID", min_value=0,
                                 max_value=len(X_val_scaled) - 1, value=0)
    current_oee = st.slider("Shift OEE % (from Digital Twin)", 0.0, 1.0, 0.72, 0.01)

    run_analysis = st.button("📊 Run Analysis", use_container_width=True)

    if run_analysis or "rca_result" in st.session_state:
        if run_analysis:
            sample_scaled = X_val_scaled.iloc[[sample_id]]
            sample_real   = pd.DataFrame(
                scaler.inverse_transform(sample_scaled), columns=feature_names
            )
            with st.spinner("Running RCA + Counterfactual Analysis…"):
                st.session_state["rca_result"]  = rca_engine.analyze_cycle(sample_real)
                st.session_state["cf_result"]   = cf_engine.analyze(sample_real)
                st.session_state["last_sample"] = sample_id

        rca = st.session_state["rca_result"]
        cf  = st.session_state["cf_result"]

        health     = rca.get('system_health_score', 100)
        n_critical = len(rca.get('critical_anomalies', []))
        tier       = cf.get('tier', 3)
        vok        = cf.get('validator_ok', False)

        st.markdown("**Live Diagnostics:**")
        st.metric("System Health", f"{health:.0f}%")
        st.metric("RCA Tier", f"Tier {tier}", cf.get('message', '')[:50])

        if n_critical:
            st.error(f"🚨 {n_critical} Critical Sensor Faults")
        elif rca.get('warnings'):
            st.warning(f"⚠️ {len(rca['warnings'])} Warnings")
        else:
            st.success("✅ Sensors within normal bounds")

        if tier < 3:
            vok_str = "✅ Confirmed" if vok else "⚠️ Unconfirmed"
            st.info(f"Counterfactual validator: {vok_str}")

with col2:
    st.markdown("#### 2. Generate Report")
    st.caption("Report is grounded in live RCA + counterfactual parameter adjustments.")

    if st.button("📄 Draft Engineer's Report", type="primary"):
        if "rca_result" not in st.session_state:
            st.warning("Click **Run Analysis** first.")
        else:
            oee_context = {"oee": current_oee}
            with st.spinner("Consulting Llama 3 Senior Engineer…"):
                report_text = llm_engine.generate_smart_report(
                    st.session_state["rca_result"],
                    oee_context,
                    cf_result=st.session_state.get("cf_result")
                )

            st.markdown("### 🤖 AI Generated Report")
            st.markdown("---")
            st.markdown(report_text)

            sid = st.session_state.get("last_sample", sample_id)
            st.download_button(
                "💾 Download Report (TXT)",
                report_text,
                f"shift_report_cycle_{sid}.txt"
            )

            if notifier.enabled:
                if st.button("📱 Send Report to Telegram"):
                    ok = notifier.send_shift_report(report_text, cycle_id=sid)
                    if ok:
                        st.success("✅ Report sent to Telegram!")
                    else:
                        st.error("❌ Telegram send failed — check token/chat_id in .env")
            else:
                st.caption("📱 Telegram not configured — add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env")
