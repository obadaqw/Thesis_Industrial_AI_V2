"""
Module: 1_🤖_Model_Forge.py
Responsibility: Logic implementation for 1_🤖_Model_Forge.
Reference: Thesis Architecture Document.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import sys
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from model_factory import train_model
import experiment_tracker
from role_manager import render_role_selector, render_access_gate

st.set_page_config(page_title="Model Forge", page_icon="🤖", layout="wide")


# LOAD CUSTOM CSS
def local_css():
    st.markdown("""
        <style>
        .metric-card {
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #444;
            text-align: center;
        }
        .stPlotlyChart {
            background-color: #262730;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)


local_css()
render_role_selector()
render_access_gate("Model Forge")

st.title("🤖 Model Forge & AutoML")
st.markdown("### Train, Benchmark, and Validate Industrial AI Models")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    model_choice = st.selectbox(
        "Select Architecture",
        ["RF", "XGB", "MLP", "SVM", "KNN", "GB", "DT"],
        index=0,
        help="RF (Random Forest) is the thesis-selected champion — "
             "highest accuracy and Waste recall, TreeSHAP compatible."
    )

    st.info("""
    **Model Guide:**
    * **RF:** 🏆 Thesis Champion — highest accuracy and Waste recall, TreeSHAP compatible.
    * **XGB:** Strong alternative for Tabular Data (High Accuracy).
    * **MLP:** Neural Network (Good for complex patterns).
    * **SVM:** Good for small datasets.
    """)

    train_btn = st.button("🚀 Train New Model", use_container_width=True, type="primary")

# --- MAIN LOGIC ---
if train_btn:
    with st.spinner(f"Training {model_choice} architecture... This may take a moment."):
        try:
            train_model(algo_name=model_choice)
            st.success(f"✅ {model_choice} Trained Successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Training Failed: {e}")

# --- DASHBOARD ---
model_path = os.path.join("models", "checkpoints", "current_model.pkl")

if os.path.exists(model_path):
    model = joblib.load(model_path)

    # LOAD TEST DATA FOR LIVE VALIDATION
    processed_dir = os.path.join("data", "processed")
    X_val = pd.read_csv(os.path.join(processed_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(processed_dir, "y_val.csv")).values.ravel()

    # FIX FOR 1-BASED INDEXING
    if hasattr(model, "classes_") and model.classes_[0] == 0 and y_val.min() == 1:
        y_val = y_val - 1

    # PREDICT
    y_pred = model.predict(X_val)

    # METRICS
    acc = np.mean(y_pred == y_val)
    from sklearn.metrics import f1_score

    f1 = f1_score(y_val, y_pred, average='weighted')

    # --- ROW 1: KPI CARDS ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model Architecture", type(model).__name__)
    with c2:
        st.metric("Validation Accuracy", f"{acc:.2%}", delta="Target: >90%")
    with c3:
        st.metric("Weighted F1-Score", f"{f1:.2f}")

    st.markdown("---")

    # --- ROW 2: DEEP DIVE VISUALS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Confusion Matrix", "📉 Feature Importance", "📑 Class Report", "🏅 Experiment History"])

    with tab1:
        st.markdown("#### Where is the model making mistakes?")
        cm = confusion_matrix(y_val, y_pred)

        # heatmap
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis",  # Fixed color scale
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=["Waste", "Acceptable", "Target", "Inefficient"],
            y=["Waste", "Acceptable", "Target", "Inefficient"]
        )
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_cm, width='stretch')

    with tab2:
        if hasattr(model, "feature_importances_"):
            st.markdown("#### Which sensors drive the decisions?")
            feat_names = joblib.load(os.path.join("models", "checkpoints", "feature_names.pkl"))

            imp_df = pd.DataFrame({
                'Sensor': feat_names,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=True).tail(15)

            fig_imp = px.bar(
                imp_df, x='Importance', y='Sensor', orientation='h',
                color='Importance', color_continuous_scale='Teal'  # FIXED: 'Cyan' -> 'Teal'
            )
            fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_imp, width='stretch')
        else:
            st.info(
                "ℹ️ This algorithm (e.g., SVM/MLP) works as a 'Black Box' and does not provide simple feature importance. Use the XAI Lab for insights.")

    with tab3:
        st.markdown("#### Detailed Performance by Class")
        report = classification_report(y_val, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.highlight_max(axis=0, color='darkgreen'))

    with tab4:
        st.markdown("#### All Training Runs — Experiment History")
        exp_df = experiment_tracker.load_df()
        if exp_df.empty:
            st.info("No experiments logged yet. Train a model to start tracking.")
        else:
            champion = experiment_tracker.get_champion()
            st.success(
                f"🏆 Champion: **{champion['algo']}** — "
                f"Val Acc: {champion['val_acc']:.2%} | "
                f"CV: {champion['cv_mean']:.4f} ± {champion['cv_std']:.4f}"
            )

            def _hl_best(val):
                if isinstance(val, float) and val == exp_df["val_acc"].max():
                    return "background-color: #003a20; font-weight: bold"
                return ""

            st.dataframe(
                exp_df.style
                    .format({"cv_mean": "{:.4f}", "cv_std": "{:.4f}",
                             "val_acc": "{:.2%}", "val_f1":  "{:.4f}"})
                    .map(_hl_best, subset=["val_acc"]),
                width='stretch', hide_index=True
            )

            fig_hist = px.bar(
                exp_df.sort_values("val_acc"),
                x="algo", y="val_acc", color="algo",
                text=exp_df.sort_values("val_acc")["val_acc"].apply(lambda v: f"{v:.2%}"),
                title="Validation Accuracy by Algorithm",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_hist.add_hline(y=0.90, line_dash="dash", line_color="white",
                               annotation_text="90% target")
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), height=350, showlegend=False
            )
            st.plotly_chart(fig_hist, width='stretch')

else:
    st.warning("⚠️ No trained model found. Click 'Train New Model' in the sidebar.")