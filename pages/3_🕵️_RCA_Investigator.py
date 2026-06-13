import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from counterfactual_rca import CounterfactualRCA
from telegram_notifier import TelegramNotifier
from role_manager import render_role_selector, render_access_gate
import cycle_store

st.set_page_config(page_title="RCA Investigator", page_icon="🕵️", layout="wide")

st.markdown("""
    <style>
    .metric-card { background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #444; }
    .tier-badge  { padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 1.1em; display: inline-block; }
    </style>
""", unsafe_allow_html=True)

render_role_selector()
render_access_gate("RCA Investigator")

st.title("🕵️ RCA Investigator — Counterfactual Analysis")
st.markdown("### 3-Tier Root-Cause Analysis with SHAP + LIME + NN-Anchoring")

CLASS_NAMES = {0: "Waste", 1: "Acceptable", 2: "Target ✅", 3: "Inefficient"}
TIER_LABELS = {0: "✅ Already Acceptable", 1: "🔵 Tier 1 — SHAP+LIME",
               2: "🟡 Tier 2 — NN-Anchored", 3: "🔴 Tier 3 — Escalation"}
TIER_HEALTH = {0: 100, 1: 75, 2: 50, 3: 0}
TIER_COLOR  = {0: "#00FFCC", 1: "#00BFFF", 2: "#FFA500", 3: "#FF4B4B"}


@st.cache_resource(show_spinner="Initializing CounterfactualRCA (training MLP validator)…")
def load_rca():
    return CounterfactualRCA(), TelegramNotifier()


@st.cache_data
def load_val_data():
    scaler        = joblib.load(os.path.join("models", "checkpoints", "scaler.pkl"))
    feature_names = joblib.load(os.path.join("models", "checkpoints", "feature_names.pkl"))
    X_val_scaled  = pd.read_csv(os.path.join("data", "processed", "X_val.csv"))
    y_val_raw     = pd.read_csv(os.path.join("data", "processed", "y_val.csv")).values.ravel()
    y_val         = y_val_raw - 1 if y_val_raw.min() > 0 else y_val_raw
    return scaler, feature_names, X_val_scaled, y_val


try:
    rca_engine, notifier = load_rca()
    scaler, feature_names, X_val_scaled, y_val = load_val_data()
except Exception as e:
    st.error(f"❌ Engine Error: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔎 Case Selection")
    sample_id = st.number_input(
        "Select Cycle ID (0-based)", min_value=0,
        max_value=len(X_val_scaled) - 1, value=0
    )

    actual_quality = int(y_val[sample_id]) + 1
    st.info(f"Actual Quality: **{actual_quality}** — {CLASS_NAMES[actual_quality-1]}")

    analyze_btn = st.button("🔍 Run Counterfactual Analysis", type="primary",
                            use_container_width=True)

    st.markdown("---")
    if notifier.enabled:
        st.success("📱 Telegram: active")
    else:
        st.caption("📱 Telegram: not configured (see .env)")

    st.markdown("---")
    st.markdown("""
    **How the 3 tiers work:**
    - 🔵 **Tier 1** — SHAP picks top-5 features; LIME gives the adjustment direction.
    - 🟡 **Tier 2** — Finds 15 nearest good-quality samples, moves toward their centroid on top-7 features.
    - 🔴 **Tier 3** — Structurally uncorrectable; escalate for manual inspection.

    **Acceptance:** P(Acceptable) + P(Target) ≥ 55 %

    **Validator:** Independent MLP (seed=7) confirms each accepted counterfactual.
    """)

# ── Main area ─────────────────────────────────────────────────────────────────
if analyze_btn:
    sample_scaled = X_val_scaled.iloc[[sample_id]]
    sample_real   = pd.DataFrame(
        scaler.inverse_transform(sample_scaled),
        columns=feature_names
    )

    with st.spinner("Running 3-tier counterfactual analysis…"):
        result = rca_engine.analyze(sample_real)

    tier    = result['tier']
    status  = result['status']
    pred    = result['prediction']
    conf    = result['confidence']
    vok     = result['validator_ok']
    adj     = result['adjustments']
    color   = TIER_COLOR[tier]
    health  = TIER_HEALTH[tier]

    # Persist cycle to history DB
    try:
        cycle_store.log_cycle(
            cycle_id=int(sample_id),
            prediction=pred,
            confidence=conf,
            rca_tier=tier,
            rca_status=status,
            cf_confidence=result.get('cf_confidence', 0.0),
            validator_ok=bool(vok)
        )
    except Exception:
        pass

    # Telegram: alert on defect + send RCA summary (tier 1, 2, or 3)
    if pred not in (1, 2):
        notifier.send_defect_alert(sample_id, pred, conf)
    if tier >= 1:
        notifier.send_rca_alert(
            sample_id, tier, adj, vok,
            result.get('cf_confidence', 0.0)
        )

    # ── KPI row ───────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("AI Prediction", f"Quality {pred+1}", CLASS_NAMES[pred])
    k2.metric("Input Confidence", f"{conf:.1%}",
              "↑ meets threshold" if conf >= 0.55 else "↓ below threshold")
    k3.metric("Health Score", f"{health}%")
    k4.metric("Tier", TIER_LABELS[tier])

    st.markdown("---")

    # ── Health gauge + tier detail ─────────────────────────────────────────
    col_gauge, col_detail = st.columns([1, 2])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health,
            title={"text": "Cycle Health"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0, 30],  "color": "#3a0000"},
                    {"range": [30, 60], "color": "#3a2000"},
                    {"range": [60, 100],"color": "#003a20"},
                ]
            }
        ))
        fig.update_layout(height=280,
                          margin=dict(l=20, r=20, t=50, b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"))
        st.plotly_chart(fig, width='stretch')

        vok_str = "✅ Confirmed" if vok else ("⚠️ Unconfirmed" if tier < 3 else "N/A")
        st.markdown(f"**Independent Validator:** {vok_str}")
        if tier > 0:
            cf_c = result.get('cf_confidence', 0)
            st.markdown(f"**CF Confidence:** {cf_c:.1%}" if cf_c else "")

    with col_detail:
        if tier == 0:
            st.success(f"✅ {result['message']}")

        elif tier in (1, 2):
            tier_name = "SHAP + LIME" if tier == 1 else f"NN-Anchored (15 neighbors)"
            st.success(f"**{TIER_LABELS[tier]}** — counterfactual found.")
            st.info(result['message'])

            st.markdown("#### 🔧 Repair Recipe (Physical Units)")
            adj_df = pd.DataFrame(adj)

            def _color_direction(val):
                if val == "↑":   return "color: #00FFCC; font-weight:bold"
                if val == "↓":   return "color: #FF8C00; font-weight:bold"
                return ""

            def _color_delta(val):
                return "color: #00FFCC" if val > 0 else ("color: #FF8C00" if val < 0 else "")

            st.dataframe(
                adj_df.style
                    .map(_color_direction, subset=["direction"])
                    .map(_color_delta,    subset=["delta"]),
                width='stretch',
                hide_index=True
            )

            csv = adj_df.to_csv(index=False).encode()
            st.download_button("💾 Download Repair Recipe",
                               csv, f"repair_recipe_cycle_{sample_id}.csv",
                               "text/csv")

        else:  # Tier 3
            st.error(f"🚨 {result['message']}")
            st.warning("Escalate to maintenance team. Log cycle data for offline analysis.")

else:
    st.info("👈 Select a cycle in the sidebar and click **Run Counterfactual Analysis**.")
    st.markdown("""
    <div style="text-align:center; color:#555; margin-top:60px;">
        <h3>Waiting for cycle selection…</h3>
        <p>The engine will attempt:</p>
        <ul style="display:inline-block; text-align:left;">
            <li>🔵 Tier 1 — SHAP selects · LIME directs (top-5 features)</li>
            <li>🟡 Tier 2 — NN-anchored centroid guidance (top-7 features)</li>
            <li>🔴 Tier 3 — Escalation if no CF found within physical bounds</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
