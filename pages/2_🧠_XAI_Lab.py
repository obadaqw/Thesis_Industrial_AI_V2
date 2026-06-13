import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from xai_engine import XAIEngine

st.set_page_config(page_title="XAI Lab", page_icon="🧠", layout="wide")

st.title("🧠 XAI Differential Diagnosis Lab")
st.markdown("### SHAP-Based Defect vs Golden Sample Comparison")


@st.cache_resource
def load_engine():
    return XAIEngine()


@st.cache_data
def load_val_data():
    processed_dir = os.path.join("data", "processed")
    X_val = pd.read_csv(os.path.join(processed_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(processed_dir, "y_val.csv")).values.ravel()
    y_adj = y_val - 1 if y_val.min() > 0 else y_val
    return X_val, y_adj


try:
    engine = load_engine()
    X_val, y_val = load_val_data()
except Exception as e:
    st.error(f"❌ Engine load failed: {e}")
    st.stop()

CLASS_NAMES = {0: "Waste", 1: "Acceptable", 2: "Target ✅", 3: "Inefficient"}

defect_indices = np.where(y_val != 2)[0]
golden_indices = np.where(y_val == 2)[0]

if len(defect_indices) == 0 or len(golden_indices) == 0:
    st.error("Not enough labelled samples in validation set.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔬 Sample Selection")
    defect_id = st.selectbox(
        "Defective Sample (cycle index)",
        defect_indices,
        format_func=lambda i: f"Cycle {i} — Quality {y_val[i]+1}: {CLASS_NAMES[y_val[i]]}"
    )
    golden_id = st.selectbox(
        "Golden Reference Sample (cycle index)",
        golden_indices,
        format_func=lambda i: f"Cycle {i} — Quality 3: Target"
    )
    run_btn = st.button("🔍 Run Differential Diagnosis", type="primary", use_container_width=True)

    st.markdown("---")
    st.info("""
    **How it works:**
    SHAP values measure each feature's push on the model's confidence
    toward the Target class. The **Delta** is the absolute difference
    between the defect sample's push and the golden sample's push —
    the largest deltas identify root-cause sensors.
    """)

# --- MAIN CONTENT ---
if run_btn:
    defect_row = X_val.iloc[[defect_id]]
    golden_row = X_val.iloc[[golden_id]]

    with st.spinner("Computing SHAP differential..."):
        report = engine.get_differential_diagnosis(defect_row, golden_row)

    # Predictions
    defect_pred = engine.model.predict(defect_row)[0]
    golden_pred = engine.model.predict(golden_row)[0]
    defect_conf = float(np.max(engine.model.predict_proba(defect_row)))
    golden_conf = float(np.max(engine.model.predict_proba(golden_row)))

    # KPI row
    c1, c2, c3 = st.columns(3)
    c1.error(f"🔴 Defect — Quality {defect_pred+1}: {CLASS_NAMES[defect_pred]}  \n**Confidence:** {defect_conf:.1%}")
    c2.success(f"🟢 Golden — Quality {golden_pred+1}: {CLASS_NAMES[golden_pred]}  \n**Confidence:** {golden_conf:.1%}")
    top_cause = report.iloc[0]['Feature']
    top_delta = report.iloc[0]['Delta_Contribution']
    c3.warning(f"🏆 Top Root Cause  \n**{top_cause}**  \nΔ = {top_delta:.4f}")

    st.markdown("---")

    top_n = report.head(10)

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        st.markdown("#### Feature SHAP Contributions — Defect vs Golden")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_n['Feature'],
            x=top_n['Defect_Impact'],
            name='Defect',
            orientation='h',
            marker_color='#FF4B4B'
        ))
        fig.add_trace(go.Bar(
            y=top_n['Feature'],
            x=top_n['Golden_Impact'],
            name='Golden (Target)',
            orientation='h',
            marker_color='#00FFCC'
        ))
        fig.update_layout(
            barmode='group',
            xaxis_title="SHAP Value (push toward Target class)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=420,
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, width='stretch')

    with col_table:
        st.markdown("#### Delta Contribution Ranking")
        display_df = report[['Feature', 'Defect_Impact', 'Golden_Impact', 'Delta_Contribution']].head(10).copy()
        for col in ['Defect_Impact', 'Golden_Impact', 'Delta_Contribution']:
            display_df[col] = display_df[col].round(4)
        st.dataframe(
            display_df.style.background_gradient(subset=['Delta_Contribution'], cmap='Reds'),
            width='stretch',
            hide_index=True
        )

    st.info("💡 Features with the highest **Delta** are the most likely root causes — they contribute very differently between the defect and the golden part.")

    # ── LIME direction panel ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🧭 LIME Adjustment Directions (for Target class)")
    st.caption("SHAP selects WHICH features matter most. LIME tells you WHICH WAY to adjust them.")

    with st.spinner("Computing LIME local explanation…"):
        try:
            lime_df = engine.get_full_lime_explanation(defect_row, label=2, num_samples=2000)
            top_lime = lime_df.head(10)

            lc1, lc2 = st.columns([2, 1])
            with lc1:
                import plotly.express as px
                fig_lime = px.bar(
                    top_lime.sort_values('LIME_Coefficient'),
                    x='LIME_Coefficient', y='Feature', orientation='h',
                    color='LIME_Coefficient',
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0,
                    labels={'LIME_Coefficient': 'LIME Coefficient (→ Target class)'}
                )
                fig_lime.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"), height=350,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_lime, width='stretch')

            with lc2:
                st.dataframe(
                    top_lime[['Feature', 'LIME_Coefficient', 'Direction']].style
                        .background_gradient(subset=['LIME_Coefficient'], cmap='RdYlGn'),
                    width='stretch', hide_index=True
                )
        except Exception as ex:
            st.warning(f"LIME explanation unavailable: {ex}")

else:
    st.info("👈 Select a defect and golden sample in the sidebar, then click **Run Differential Diagnosis**.")
    st.markdown("""
    <div style="text-align:center; color:#666; margin-top:60px;">
        <h3>Waiting for Sample Selection...</h3>
        <p>The engine will compare:</p>
        <ul style="display:inline-block; text-align:left;">
            <li>🔴 SHAP contributions of the <b>defective</b> part</li>
            <li>🟢 SHAP contributions of the <b>golden</b> reference</li>
            <li>📊 <b>Delta</b> = absolute difference (root cause signal)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
