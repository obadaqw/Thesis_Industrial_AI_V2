import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import yaml
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

st.set_page_config(page_title="ISO 9001 Dashboard", page_icon="📊", layout="wide")

st.title("📊 ISO 9001 Quality Management Dashboard")
st.markdown("### Statistical Process Control · Process Capability · Non-Conformance Analysis")

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scaler        = joblib.load(os.path.join(base, "models", "checkpoints", "scaler.pkl"))
    feature_names = joblib.load(os.path.join(base, "models", "checkpoints", "feature_names.pkl"))
    model         = joblib.load(os.path.join(base, "models", "checkpoints", "current_model.pkl"))

    X_val_scaled = pd.read_csv(os.path.join(base, "data", "processed", "X_val.csv"))
    y_val_raw    = pd.read_csv(os.path.join(base, "data", "processed", "y_val.csv")).values.ravel()
    y_val        = y_val_raw - 1 if y_val_raw.min() > 0 else y_val_raw

    X_train_scaled = pd.read_csv(os.path.join(base, "data", "processed", "X_train.csv"))
    y_train_raw    = pd.read_csv(os.path.join(base, "data", "processed", "y_train.csv")).values.ravel()
    y_train        = y_train_raw - 1 if y_train_raw.min() > 0 else y_train_raw

    # Physical units (inverse-transform)
    X_val_real   = pd.DataFrame(scaler.inverse_transform(X_val_scaled),   columns=feature_names)
    X_train_real = pd.DataFrame(scaler.inverse_transform(X_train_scaled), columns=feature_names)

    # AI predictions on validation set
    y_pred = model.predict(X_val_scaled)

    # Constraints (USL / LSL)
    with open(os.path.join(base, "config", "constraints.yaml")) as f:
        constraints = yaml.safe_load(f)

    return X_val_real, y_val, y_pred, X_train_real, y_train, feature_names, constraints


try:
    X_val, y_val, y_pred, X_train, y_train, feature_names, constraints = load_data()
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

CLASS_NAMES  = {0: "Waste", 1: "Acceptable", 2: "Target", 3: "Inefficient"}
CLASS_COLORS = {0: "#FF4B4B", 1: "#FFA500", 2: "#00FFCC", 3: "#9B59B6"}

# ── Capability helpers ────────────────────────────────────────────────────────

def compute_capability(series, usl, lsl):
    mu  = series.mean()
    sig = series.std(ddof=1)
    if sig == 0:
        return np.nan, np.nan
    cp  = (usl - lsl) / (6 * sig)
    cpk = min((usl - mu) / (3 * sig), (mu - lsl) / (3 * sig))
    return round(cp, 3), round(cpk, 3)

def cpk_color(val):
    if pd.isna(val) or val < 1.0:
        return "#FF4B4B"
    if val < 1.33:
        return "#FFA500"
    return "#00CC88"

def cpk_status(val):
    if pd.isna(val) or val < 1.0:
        return "❌ Not Capable"
    if val < 1.33:
        return "⚠️ Marginal"
    return "✅ Capable"

# ── Build capability table ────────────────────────────────────────────────────

cap_rows = []
for feat in feature_names:
    if feat not in constraints:
        continue
    lsl = constraints[feat]["min"]
    usl = constraints[feat]["max"]
    cp, cpk = compute_capability(X_val[feat], usl, lsl)
    cap_rows.append({
        "Feature":  feat,
        "LSL":      lsl,
        "USL":      usl,
        "Mean":     round(X_val[feat].mean(), 4),
        "Std":      round(X_val[feat].std(ddof=1), 4),
        "Cp":       cp,
        "Cpk":      cpk,
        "Status":   cpk_status(cpk),
    })

cap_df = pd.DataFrame(cap_rows)

# ── Overall KPIs ──────────────────────────────────────────────────────────────

n_total   = len(y_val)
n_target  = int((y_val == 2).sum())
n_accept  = int((y_val == 1).sum())
n_waste   = int((y_val == 0).sum())
n_ineff   = int((y_val == 3).sum())
fpy       = (n_target + n_accept) / n_total
defect_rt = (n_waste + n_ineff) / n_total
n_capable = int((cap_df["Cpk"] >= 1.33).sum())
n_feat    = len(cap_df)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("First Pass Yield",  f"{fpy:.1%}")
col2.metric("Defect Rate",        f"{defect_rt:.1%}", delta=f"-{defect_rt:.1%}", delta_color="inverse")
col3.metric("Target-Quality Parts", f"{n_target}/{n_total}")
col4.metric("Capable Features (Cpk≥1.33)", f"{n_capable}/{n_feat}")
col5.metric("Total Cycles Analysed", n_total)

st.markdown("---")

# ── Three tabs ────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🎯 Process Capability (Cp / Cpk)",
    "📈 SPC Control Charts",
    "📋 Non-Conformance Analysis"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Process Capability
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("#### Process Capability Indices — All 13 Features")
    st.caption(
        "**Cp** = spread capability (USL−LSL)/(6σ). "
        "**Cpk** = centred capability, accounts for process mean offset. "
        "Industry standard: Cpk ≥ 1.33 = Capable, ≥ 1.67 = Excellent."
    )

    # Colour-coded Cpk bar chart
    fig_cpk = go.Figure()
    colors = [cpk_color(v) for v in cap_df["Cpk"]]
    fig_cpk.add_trace(go.Bar(
        x=cap_df["Feature"], y=cap_df["Cpk"],
        marker_color=colors, name="Cpk",
        text=cap_df["Cpk"].astype(str), textposition="outside"
    ))
    fig_cpk.add_hline(y=1.33, line_dash="dash", line_color="#00CC88",
                      annotation_text="Capable (1.33)", annotation_position="right")
    fig_cpk.add_hline(y=1.0,  line_dash="dot",  line_color="#FFA500",
                      annotation_text="Min acceptable (1.0)", annotation_position="right")
    fig_cpk.update_layout(
        xaxis_tickangle=-35, yaxis_title="Cpk",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=400,
        margin=dict(l=10, r=80, t=20, b=10), showlegend=False
    )
    st.plotly_chart(fig_cpk, width='stretch')

    # Detailed table with colour styling
    def _style_cpk(val):
        return f"color: {cpk_color(val)}; font-weight: bold"

    def _style_status(val):
        if "✅" in str(val): return "color: #00CC88"
        if "⚠️" in str(val): return "color: #FFA500"
        return "color: #FF4B4B"

    st.dataframe(
        cap_df.style
            .map(_style_cpk,     subset=["Cp", "Cpk"])
            .map(_style_status,  subset=["Status"]),
        width='stretch', hide_index=True
    )

    # Gauge summary
    pct_capable = n_capable / n_feat * 100
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct_capable,
        title={"text": "Process Capability Score (% features Cpk≥1.33)"},
        delta={"reference": 75},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#00CC88" if pct_capable >= 75 else "#FFA500"},
            "steps": [
                {"range": [0, 50],   "color": "#3a0000"},
                {"range": [50, 75],  "color": "#3a2000"},
                {"range": [75, 100], "color": "#003a20"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "value": 75}
        }
    ))
    fig_gauge.update_layout(
        height=280, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), margin=dict(l=20, r=20, t=60, b=10)
    )
    gc1, gc2, gc3 = st.columns([1, 2, 1])
    with gc2:
        st.plotly_chart(fig_gauge, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SPC Control Charts
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("#### Statistical Process Control — X-bar Chart with 3σ Limits")
    st.caption(
        "Control limits are derived from the **training distribution** (μ ± 3σ). "
        "Red markers = out-of-control signals (beyond UCL/LCL or 8 consecutive "
        "points on same side of centreline)."
    )

    # Feature selector
    selected_feat = st.selectbox("Select Feature", feature_names, key="spc_feat")

    train_mu  = X_train[selected_feat].mean()
    train_sig = X_train[selected_feat].std(ddof=1)
    ucl = train_mu + 3 * train_sig
    lcl = train_mu - 3 * train_sig
    u1  = train_mu + train_sig
    l1  = train_mu - train_sig
    u2  = train_mu + 2 * train_sig
    l2  = train_mu - 2 * train_sig

    # Constraints for spec limits
    lsl_val = constraints.get(selected_feat, {}).get("min")
    usl_val = constraints.get(selected_feat, {}).get("max")

    values = X_val[selected_feat].values
    cycles = np.arange(len(values))

    # Out-of-control detection: beyond 3σ or 8 consecutive same-side of CL
    ooc_mask = (values > ucl) | (values < lcl)
    same_side = np.sign(values - train_mu)
    run_ooc = np.zeros(len(values), dtype=bool)
    run_len = 0
    for i, s in enumerate(same_side):
        if i == 0:
            run_len = 1
        elif s == same_side[i - 1]:
            run_len += 1
        else:
            run_len = 1
        if run_len >= 8:
            run_ooc[max(0, i - 7): i + 1] = True
    ooc_any = ooc_mask | run_ooc

    fig_spc = go.Figure()

    # Zone bands (faint)
    fig_spc.add_hrect(y0=u2, y1=ucl, fillcolor="rgba(255,75,75,0.08)", line_width=0)
    fig_spc.add_hrect(y0=lcl, y1=l2, fillcolor="rgba(255,75,75,0.08)", line_width=0)
    fig_spc.add_hrect(y0=l1, y1=u1, fillcolor="rgba(0,200,136,0.08)", line_width=0)

    # Reference lines
    for y_val_line, name, color, dash in [
        (ucl,      "UCL (+3σ)", "#FF4B4B", "dash"),
        (u2,       "+2σ",       "#FFA500", "dot"),
        (train_mu, "CL (μ)",    "#FFFFFF",  "solid"),
        (l2,       "−2σ",       "#FFA500", "dot"),
        (lcl,      "LCL (−3σ)", "#FF4B4B", "dash"),
    ]:
        fig_spc.add_hline(y=y_val_line, line_dash=dash, line_color=color,
                          annotation_text=f"{name}={y_val_line:.3f}",
                          annotation_position="right",
                          annotation_font_size=10)

    # Spec limits (if available)
    if usl_val is not None:
        fig_spc.add_hline(y=usl_val, line_dash="longdash", line_color="#9B59B6",
                          annotation_text=f"USL={usl_val}", annotation_position="left",
                          annotation_font_size=10)
    if lsl_val is not None:
        fig_spc.add_hline(y=lsl_val, line_dash="longdash", line_color="#9B59B6",
                          annotation_text=f"LSL={lsl_val}", annotation_position="left",
                          annotation_font_size=10)

    # In-control points
    in_ctrl = ~ooc_any
    fig_spc.add_trace(go.Scatter(
        x=cycles[in_ctrl], y=values[in_ctrl],
        mode="lines+markers", name="In Control",
        marker=dict(color="#00FFCC", size=5),
        line=dict(color="#00FFCC", width=1)
    ))
    # Out-of-control points
    if ooc_any.any():
        fig_spc.add_trace(go.Scatter(
            x=cycles[ooc_any], y=values[ooc_any],
            mode="markers", name="Out of Control",
            marker=dict(color="#FF4B4B", size=9, symbol="x")
        ))

    cp_f, cpk_f = compute_capability(pd.Series(values), usl_val or ucl, lsl_val or lcl)
    fig_spc.update_layout(
        title=dict(
            text=f"{selected_feat}  |  Cpk={cpk_f}  |  OOC points: {ooc_any.sum()}",
            font=dict(color="white", size=14)
        ),
        xaxis_title="Cycle Index",
        yaxis_title=selected_feat,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=450,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=10, r=80, t=50, b=10)
    )
    st.plotly_chart(fig_spc, width='stretch')

    # OOC summary row
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Cycles", len(values))
    mc2.metric("Out-of-Control", int(ooc_any.sum()),
               delta=f"{ooc_any.mean():.1%}", delta_color="inverse")
    mc3.metric("Cpk (SPC basis)", cpk_f)
    mc4.metric("UCL / LCL", f"{ucl:.3f} / {lcl:.3f}")

    # OOC summary across all features
    st.markdown("---")
    st.markdown("#### Out-of-Control Summary — All Features")
    ooc_rows = []
    for feat in feature_names:
        mu_t = X_train[feat].mean()
        sg_t = X_train[feat].std(ddof=1)
        v = X_val[feat].values
        ooc_n = int(((v > mu_t + 3 * sg_t) | (v < mu_t - 3 * sg_t)).sum())
        ooc_rows.append({"Feature": feat,
                         "OOC Signals": ooc_n,
                         "OOC Rate": f"{ooc_n/len(v):.1%}"})
    ooc_df = pd.DataFrame(ooc_rows).sort_values("OOC Signals", ascending=False)
    st.dataframe(ooc_df, width='stretch', hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Non-Conformance Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("#### Non-Conformance Analysis")

    nc1, nc2 = st.columns([1, 1])

    with nc1:
        st.markdown("##### Quality Class Distribution (Actual Labels)")
        counts = [int((y_val == c).sum()) for c in range(4)]
        fig_pie = go.Figure(go.Pie(
            labels=[CLASS_NAMES[c] for c in range(4)],
            values=counts,
            marker_colors=[CLASS_COLORS[c] for c in range(4)],
            hole=0.45,
            textinfo="label+percent+value"
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
            height=340, margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig_pie, width='stretch')

    with nc2:
        st.markdown("##### AI Model Predictions vs Actual")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3])
        fig_cm = px.imshow(
            cm,
            x=[f"Pred: {CLASS_NAMES[c]}" for c in range(4)],
            y=[f"Act: {CLASS_NAMES[c]}"  for c in range(4)],
            color_continuous_scale="Teal",
            text_auto=True,
            aspect="auto"
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
            height=340, margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_cm, width='stretch')

    # FPY trend (rolling window)
    st.markdown("---")
    st.markdown("##### First Pass Yield — Rolling Trend")

    window = st.slider("Rolling window (cycles)", 10, 60, 20, key="fpy_window")
    good   = ((y_val == 1) | (y_val == 2)).astype(float)
    rolling_fpy = pd.Series(good).rolling(window=window, min_periods=1).mean() * 100

    target_line = fpy * 100

    fig_fpy = go.Figure()
    fig_fpy.add_trace(go.Scatter(
        x=np.arange(len(rolling_fpy)), y=rolling_fpy,
        mode="lines", name=f"Rolling FPY ({window}-cycle)",
        line=dict(color="#00BFFF", width=2), fill="tozeroy",
        fillcolor="rgba(0,191,255,0.1)"
    ))
    fig_fpy.add_hline(y=target_line, line_dash="dash", line_color="#00FFCC",
                      annotation_text=f"Overall FPY {target_line:.1f}%",
                      annotation_position="right")
    fig_fpy.add_hline(y=80, line_dash="dot", line_color="#FFA500",
                      annotation_text="Target 80%", annotation_position="right")
    fig_fpy.update_layout(
        xaxis_title="Cycle Index", yaxis_title="First Pass Yield (%)",
        yaxis=dict(range=[0, 105]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=350,
        margin=dict(l=10, r=80, t=20, b=10)
    )
    st.plotly_chart(fig_fpy, width='stretch')

    # Non-conformance per class breakdown table
    st.markdown("---")
    st.markdown("##### Non-Conformance Register")
    nc_data = []
    for c in range(4):
        n = int((y_val == c).sum())
        nc_data.append({
            "Class":         f"Q{c+1}: {CLASS_NAMES[c]}",
            "Count":         n,
            "Rate":          f"{n/n_total:.2%}",
            "Conforming":    "✅ YES" if c in (1, 2) else "❌ NO",
            "ISO 9001 Note": (
                "Prime quality — target outcome"            if c == 2 else
                "Acceptable — within tolerance"             if c == 1 else
                "Non-conformance — scrap / rework required" if c == 0 else
                "Sub-optimal — process adjustment advised"
            )
        })
    st.dataframe(pd.DataFrame(nc_data), width='stretch', hide_index=True)

    # Model accuracy for non-conformance detection
    st.markdown("---")
    precision_nc = int(np.sum((y_pred == y_val) & np.isin(y_val, [0, 3]))) / max(1, int(np.isin(y_val, [0, 3]).sum()))
    recall_nc    = int(np.sum((y_pred == y_val) & np.isin(y_val, [0, 3]))) / max(1, int(np.isin(y_val, [0, 3]).sum()))
    a1, a2, a3 = st.columns(3)
    a1.metric("Overall Validation Accuracy",
              f"{(y_pred == y_val).mean():.1%}")
    a2.metric("Non-Conformance Detection Rate",
              f"{precision_nc:.1%}")
    a3.metric("Conforming Part Yield (AI)",
              f"{(np.isin(y_pred, [1,2])).mean():.1%}")
