"""
Module: 4_🏭_Digital_Twin.py
Responsibility: Logic implementation for 4_🏭_Digital_Twin.
Reference: Thesis Architecture Document.
"""
import streamlit as st
import time
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from digital_twin import DigitalTwin
from xai_engine import XAIEngine
from telegram_notifier import TelegramNotifier
from role_manager import render_role_selector, render_access_gate

st.set_page_config(page_title="Digital Twin", page_icon="🏭", layout="wide")

render_role_selector()
render_access_gate("Digital Twin")

st.title("🏭 Digital Twin Simulation")
st.markdown("### Live OEE & Production State Monitoring")


# --- INITIALIZE ---
@st.cache_resource
def load_twin():
    return DigitalTwin(), XAIEngine(), TelegramNotifier()


twin_engine, xai_engine, notifier = load_twin()

# Load Data for Simulation Stream
data_path = os.path.join("data", "processed", "X_test.csv")
if os.path.exists(data_path):
    stream_data = pd.read_csv(data_path)
else:
    st.error("No Data found. Please check data/processed/X_test.csv")
    st.stop()

# --- CONTROLS ---
col_ctrl, col_metrics = st.columns([1, 4])

with col_ctrl:
    st.markdown("#### ⚙️ Simulation Control")
    is_running = st.toggle("🔴 RUN TWIN", value=False)
    speed = st.slider("Simulation Speed (sec)", 0.1, 2.0, 0.5)
    oee_threshold = st.slider("OEE Alert Threshold", 0.5, 0.95, 0.70, 0.05)

    if st.button("Reset Machine State"):
        twin_engine.reset_machine()
        st.session_state.pop("oee_alerted", None)
        st.success("Reset!")

    if notifier.enabled:
        st.success("📱 Telegram: active")
    else:
        st.caption("📱 Telegram: not configured")

# --- LIVE DASHBOARD ---
placeholder = st.empty()

if is_running:
    # Simulation Loop
    for i in range(len(stream_data)):
        if not is_running: break

        # 1. Get Cycle Data
        row = stream_data.iloc[[i]]

        # 2. AI Predicts Quality
        pred = xai_engine.model.predict(row)[0]

        # 3. Update Twin State
        state = twin_engine.update_metrics(pred)

        # 3b. Telegram OEE alert — fire once when OEE first drops below threshold
        if (state['total_cycles'] > 10 and
                state['oee'] < oee_threshold and
                not st.session_state.get("oee_alerted", False)):
            notifier.send_oee_alert(
                state['oee'], oee_threshold,
                state['total_cycles'], state['good_cycles']
            )
            st.session_state["oee_alerted"] = True
        elif state['oee'] >= oee_threshold:
            st.session_state["oee_alerted"] = False

        # 4. Render UI
        with placeholder.container():
            # Top Row: OEE Big Numbers
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("OEE Score", f"{state['oee']:.1%}")
            m2.metric("Availability", f"{state['availability']:.1%}")
            m3.metric("Performance", f"{state['performance']:.1%}")
            m4.metric("Quality", f"{state['quality']:.1%}")

            # Status Bar (model class 2 = original quality 3 = Target)
            if pred == 2:
                st.success(f"Cycle #{i}: PRODUCING TARGET PARTS (Original Quality 3)")
            else:
                st.error(f"Cycle #{i}: DEFECT PRODUCED (Model Class {pred} → Original Quality {pred+1})")

            # Production Counters
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.info(f"Total Cycles: {state['total_cycles']}")
            c2.success(f"Good Parts: {state['good_cycles']}")
            c3.error(f"Scrap Parts: {state['bad_cycles']}")

        time.sleep(speed)