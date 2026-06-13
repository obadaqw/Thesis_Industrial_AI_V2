import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
import cycle_store
from role_manager import render_role_selector, render_access_gate

st.set_page_config(page_title="Cycle History", page_icon="📋", layout="wide")

render_role_selector()
render_access_gate("Cycle History")

st.title("📋 Cycle History — ISO 9001 Traceability Log")
st.markdown("### Full audit trail of every RCA analysis run")

# ── Load data ─────────────────────────────────────────────────────────────────
stats = cycle_store.get_stats()
df    = cycle_store.get_recent(n=500)

CLASS_NAMES  = {0: "Waste", 1: "Acceptable", 2: "Target ✅", 3: "Inefficient"}
TIER_LABELS  = {0: "Already OK", 1: "Tier 1 — SHAP+LIME",
                2: "Tier 2 — NN-Anchored", 3: "Tier 3 — Escalation"}

if df.empty:
    st.info(
        "No cycles logged yet. Run an analysis in the **RCA Investigator** "
        "to start populating the history."
    )
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Cycles Logged", stats["total"])
k2.metric("Good Cycles (Acc+Target)", stats["good"])
k3.metric("First Pass Yield", f"{stats['fpy']:.1%}")
k4.metric("Tier 3 Escalations", stats["tier_counts"].get(3, 0))
k5.metric("Validator Confirmed",
          int(df["validator_ok"].sum()) if "validator_ok" in df.columns else "—")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Overview Charts", "📄 Cycle Log", "🗑️ Manage"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Charts
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        # FPY over time (rolling 10)
        df_sorted = df.sort_values("id").copy()
        df_sorted["good_flag"] = df_sorted["prediction"].isin([1, 2]).astype(int)
        df_sorted["rolling_fpy"] = (
            df_sorted["good_flag"].rolling(10, min_periods=1).mean()
        )
        fig_fpy = go.Figure()
        fig_fpy.add_trace(go.Scatter(
            x=df_sorted["id"], y=df_sorted["rolling_fpy"],
            mode="lines", name="Rolling FPY (10)",
            line=dict(color="#00FFCC", width=2)
        ))
        fig_fpy.add_hline(y=0.95, line_dash="dash", line_color="#FFA500",
                          annotation_text="95% target")
        fig_fpy.update_layout(
            title="First Pass Yield — Rolling 10 Cycles",
            xaxis_title="Log Entry", yaxis_title="FPY",
            yaxis=dict(range=[0, 1.05]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), height=320,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_fpy, width='stretch')

    with col_right:
        # Tier distribution donut
        tier_df = (
            df["rca_tier"]
            .map(TIER_LABELS)
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Tier", "rca_tier": "Count",
                             "count": "Count"})
        )
        fig_tier = px.pie(
            tier_df, names="rca_tier", values="Count",
            hole=0.45,
            color_discrete_sequence=["#00CC88", "#00BFFF", "#FFA500", "#FF4B4B"]
        )
        fig_tier.update_layout(
            title="RCA Tier Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), height=320,
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_tier, width='stretch')

    # Confidence distribution histogram
    fig_conf = px.histogram(
        df, x="confidence", nbins=30,
        color_discrete_sequence=["#00BFFF"],
        labels={"confidence": "Model Confidence"},
        title="Confidence Distribution Across Logged Cycles"
    )
    fig_conf.add_vline(x=0.55, line_dash="dash", line_color="#FFA500",
                       annotation_text="CF threshold (0.55)")
    fig_conf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=280,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_conf, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Raw log
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Recent 500 Cycles (newest first)")

    display = df.copy()
    display["Quality"] = display["prediction"].map(CLASS_NAMES)
    display["Tier"]    = display["rca_tier"].map(TIER_LABELS)
    display["Validator"] = display["validator_ok"].apply(
        lambda v: "✅" if v else "—"
    )
    display["confidence"]    = display["confidence"].round(3)
    display["cf_confidence"] = display["cf_confidence"].round(3)

    show_cols = ["id", "timestamp", "cycle_id", "Quality", "confidence",
                 "Tier", "rca_status", "cf_confidence", "Validator"]

    def _color_quality(val):
        colors = {
            "Target ✅":   "color: #00FFCC; font-weight:bold",
            "Acceptable":  "color: #00CC88",
            "Waste":       "color: #FF4B4B; font-weight:bold",
            "Inefficient": "color: #FFA500",
        }
        return colors.get(val, "")

    def _color_tier(val):
        if "Tier 3" in str(val): return "color: #FF4B4B; font-weight:bold"
        if "Tier 2" in str(val): return "color: #FFA500"
        if "Tier 1" in str(val): return "color: #00BFFF"
        return "color: #00CC88"

    st.dataframe(
        display[show_cols].style
            .map(_color_quality, subset=["Quality"])
            .map(_color_tier,    subset=["Tier"]),
        width='stretch', hide_index=True
    )

    csv = display[show_cols].to_csv(index=False).encode()
    st.download_button("💾 Export to CSV", csv, "cycle_history.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Manage
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.warning("**Danger zone** — clearing the log cannot be undone.")
    if st.button("🗑️ Clear All Cycle History", type="primary"):
        cycle_store.clear_history()
        st.success("Cycle history cleared.")
        st.rerun()
