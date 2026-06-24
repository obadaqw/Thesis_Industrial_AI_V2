import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from drift_detector import DriftDetector, PSI_MODERATE, PSI_CRITICAL, psi_emoji
from role_manager import render_role_selector, render_access_gate
from telegram_notifier import TelegramNotifier

st.set_page_config(page_title="Drift Monitor", page_icon="📡", layout="wide")

render_role_selector()
render_access_gate("Drift Monitor")

st.title("📡 Drift Monitor — Statistical Process Control")
st.markdown("### Population Stability Index · Page-Hinkley Concept Drift · Distribution Overlay")


@st.cache_data
def load_data():
    base      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mdl_dir   = os.path.join(base, "models", "checkpoints")
    proc_dir  = os.path.join(base, "data", "processed")

    scaler        = joblib.load(os.path.join(mdl_dir, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(mdl_dir, "feature_names.pkl"))
    model         = joblib.load(os.path.join(mdl_dir, "current_model.pkl"))

    X_train = pd.read_csv(os.path.join(proc_dir, "X_train.csv"))
    X_val   = pd.read_csv(os.path.join(proc_dir, "X_val.csv"))
    y_val_r = pd.read_csv(os.path.join(proc_dir, "y_val.csv")).values.ravel()
    y_val   = y_val_r - 1 if y_val_r.min() > 0 else y_val_r

    # Confidence stream: P(Acceptable) + P(Target) for each val cycle
    probas     = model.predict_proba(X_val)
    confidence = (probas[:, 1] + probas[:, 2]).tolist()

    return scaler, feature_names, model, X_train, X_val, y_val, confidence


@st.cache_resource
def load_notifier():
    return TelegramNotifier()


try:
    scaler, feature_names, model, X_train, X_val, y_val, confidence = load_data()
    notifier = load_notifier()
    detector = DriftDetector(X_train, feature_names)
except Exception as e:
    st.error(f"Load failed: {e}")
    st.stop()

# ── Compute PSI ───────────────────────────────────────────────────────────────
psi_df = detector.compute_all_psi(X_val)
overall = detector.overall_status(psi_df)
n_critical = int((psi_df["PSI"] >= PSI_CRITICAL).sum())
n_moderate = int(((psi_df["PSI"] >= PSI_MODERATE) & (psi_df["PSI"] < PSI_CRITICAL)).sum())

# ── Page-Hinkley on confidence stream ────────────────────────────────────────
ph_points = detector.page_hinkley(np.array(confidence))

# ── Top KPIs ──────────────────────────────────────────────────────────────────
STATUS_COLOR = {"stable": "#00CC88", "moderate": "#FFA500", "critical": "#FF4B4B"}
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Overall Status",         overall.upper())
k2.metric("Critical Features (PSI>0.20)", n_critical)
k3.metric("Moderate Features",       n_moderate)
k4.metric("Stable Features",         len(feature_names) - n_critical - n_moderate)
k5.metric("Concept Drift Alarms",    len(ph_points))

if overall == "critical":
    st.error(f"🚨 **{n_critical} features have significant distribution drift — model retraining recommended.**")
    if notifier.enabled:
        for _, row in psi_df[psi_df["PSI"] >= PSI_CRITICAL].iterrows():
            notifier.send_iso_alert(row["Feature"], row["PSI"], threshold=PSI_CRITICAL)
elif overall == "moderate":
    st.warning("⚠️ Moderate drift detected — increase monitoring frequency.")
else:
    st.success("✅ All feature distributions are stable.")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 PSI Dashboard", "📈 Distribution Overlay", "🔔 Concept Drift"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PSI Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Population Stability Index — All Features")
    st.caption("Reference = training set · Current = validation batch")

    color_map = {
        row["Feature"]: (
            "#FF4B4B" if row["PSI"] >= PSI_CRITICAL else
            "#FFA500" if row["PSI"] >= PSI_MODERATE else
            "#00CC88"
        )
        for _, row in psi_df.iterrows()
    }

    fig_psi = go.Figure()
    fig_psi.add_trace(go.Bar(
        x=psi_df["Feature"], y=psi_df["PSI"],
        marker_color=[color_map[f] for f in psi_df["Feature"]],
        text=psi_df["PSI"].round(3).astype(str), textposition="outside"
    ))
    fig_psi.add_hline(y=PSI_CRITICAL, line_dash="dash", line_color="#FF4B4B",
                      annotation_text="Critical (0.20)", annotation_position="right")
    fig_psi.add_hline(y=PSI_MODERATE, line_dash="dot",  line_color="#FFA500",
                      annotation_text="Moderate (0.10)", annotation_position="right")
    fig_psi.update_layout(
        xaxis_tickangle=-35, yaxis_title="PSI",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=400,
        margin=dict(l=10, r=80, t=20, b=10), showlegend=False
    )
    st.plotly_chart(fig_psi, width='stretch')

    # PSI table
    display_psi = psi_df.copy()

    def _psi_style(val):
        if val >= PSI_CRITICAL: return "color: #FF4B4B; font-weight: bold"
        if val >= PSI_MODERATE: return "color: #FFA500; font-weight: bold"
        return "color: #00CC88"

    st.dataframe(
        display_psi.style.map(_psi_style, subset=["PSI"]),
        width='stretch', hide_index=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Distribution Overlay
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Feature Distribution: Training vs Current Batch")

    feat = st.selectbox("Select Feature", feature_names, key="dist_feat")
    train_vals = X_train[feat].values
    curr_vals  = X_val[feat].values

    x_range = np.linspace(
        min(train_vals.min(), curr_vals.min()),
        max(train_vals.max(), curr_vals.max()),
        200
    )
    kde_train = gaussian_kde(train_vals)(x_range)
    kde_curr  = gaussian_kde(curr_vals)(x_range)

    psi_val = detector.feature_psi(feat, X_val)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Scatter(
        x=x_range, y=kde_train, fill="tozeroy",
        name="Training (Reference)", line=dict(color="#00BFFF", width=2),
        fillcolor="rgba(0,191,255,0.15)"
    ))
    fig_dist.add_trace(go.Scatter(
        x=x_range, y=kde_curr, fill="tozeroy",
        name="Validation (Current)", line=dict(color="#FF4B4B", width=2),
        fillcolor="rgba(255,75,75,0.15)"
    ))
    fig_dist.update_layout(
        title=dict(
            text=f"{feat}  |  PSI = {psi_val:.4f}  |  {psi_emoji(psi_val)}",
            font=dict(color="white", size=13)
        ),
        xaxis_title=feat, yaxis_title="Density",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=420,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=20, t=50, b=10)
    )
    st.plotly_chart(fig_dist, width='stretch')

    # Stats comparison
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Train μ",  f"{train_vals.mean():.4f}")
    sc2.metric("Val μ",    f"{curr_vals.mean():.4f}",
               delta=f"{curr_vals.mean()-train_vals.mean():.4f}")
    sc3.metric("Train σ",  f"{train_vals.std():.4f}")
    sc4.metric("Val σ",    f"{curr_vals.std():.4f}",
               delta=f"{curr_vals.std()-train_vals.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Concept Drift (Page-Hinkley)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Concept Drift — Page-Hinkley Test on Model Confidence Stream")
    st.caption(
        "Confidence = P(Acceptable) + P(Target) per cycle. "
        "Page-Hinkley detects when the cumulative mean shift exceeds the threshold, "
        "signalling that the model's output distribution has changed."
    )

    cycles = np.arange(len(confidence))
    conf_arr = np.array(confidence)
    rolling_mean = pd.Series(conf_arr).rolling(20, min_periods=1).mean().values

    fig_ph = go.Figure()
    fig_ph.add_trace(go.Scatter(
        x=cycles, y=conf_arr,
        mode="lines", name="Confidence per cycle",
        line=dict(color="#00BFFF", width=1), opacity=0.5
    ))
    fig_ph.add_trace(go.Scatter(
        x=cycles, y=rolling_mean,
        mode="lines", name="Rolling mean (20 cycles)",
        line=dict(color="#00FFCC", width=2)
    ))
    if ph_points:
        ph_arr = np.array(ph_points)
        fig_ph.add_trace(go.Scatter(
            x=ph_arr, y=conf_arr[ph_arr],
            mode="markers", name="Drift alarm",
            marker=dict(color="#FF4B4B", size=12, symbol="x")
        ))
    fig_ph.add_hline(y=0.55, line_dash="dash", line_color="#FFA500",
                     annotation_text="CF threshold (0.55)", annotation_position="right")

    fig_ph.update_layout(
        xaxis_title="Cycle Index", yaxis_title="P(Acceptable) + P(Target)",
        yaxis=dict(range=[0, 1.05]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=400,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=80, t=20, b=10)
    )
    st.plotly_chart(fig_ph, width='stretch')

    ph1, ph2, ph3 = st.columns(3)
    ph1.metric("Total Cycles",      len(confidence))
    ph2.metric("Drift Alarms",      len(ph_points))
    ph3.metric("Mean Confidence",   f"{np.mean(confidence):.3f}")

    if ph_points:
        st.warning(
            f"⚠️ Page-Hinkley detected {len(ph_points)} drift alarm(s) "
            f"at cycles: {ph_points[:10]}{'...' if len(ph_points) > 10 else ''}. "
            f"Consider retraining the model."
        )
    else:
        st.success("✅ No concept drift detected in the confidence stream.")
