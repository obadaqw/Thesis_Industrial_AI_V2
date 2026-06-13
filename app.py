import streamlit as st
import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── First-boot initializer ────────────────────────────────────────────────────
# Runs data pipeline + trains RF champion if checkpoints are missing.
# Executes before any st.* call so it's safe to block here.
def _bootstrap():
    _root = os.path.dirname(os.path.abspath(__file__))
    _model = os.path.join(_root, "models", "checkpoints", "current_model.pkl")
    if os.path.exists(_model):
        return
    _src = os.path.join(_root, "src")
    print("⚙️  First boot — running data pipeline...", flush=True)
    subprocess.run(
        [sys.executable, os.path.join(_src, "data_pipeline.py")],
        check=True, cwd=_root
    )
    print("⚙️  First boot — training RF champion...", flush=True)
    subprocess.run(
        [sys.executable, os.path.join(_src, "model_factory.py"), "--algo", "RF"],
        check=True, cwd=_root
    )
    print("✅  First boot complete.", flush=True)

_bootstrap()

# Start background API server (non-blocking daemon thread)
try:
    from platform_server import start_background
    start_background(port=8502)
except Exception:
    pass

# 1. PAGE CONFIGURATION (Must be the first command)
st.set_page_config(
    page_title="Digital Shadow V2",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. LOAD CUSTOM CSS (Dark Theme)
def load_css():
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback CSS if file missing
        st.markdown("""
        <style>
            .stApp { background-color: #0E1117; color: #FAFAFA; }
            .stSidebar { background-color: #262730; }
            h1, h2, h3 { color: #00FFCC !important; font-family: 'Roboto', sans-serif; }
            .stButton>button { border-radius: 5px; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)


load_css()

# 3. SIDEBAR — role selector + status
try:
    from role_manager import render_role_selector
    render_role_selector()
except Exception:
    pass

st.sidebar.image("https://img.icons8.com/color/96/000000/industrial-robot.png", width=80)
st.sidebar.title("🏭 Digital Shadow")
st.sidebar.markdown("---")
st.sidebar.info("**Status:** System Online 🟢")
st.sidebar.caption("API: http://localhost:8502/docs")

# 4. MAIN WELCOME SCREEN
st.title("🏭 Intelligent Quality Control Center")
st.markdown("### Master's Thesis: Explainable AI for Injection Molding")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **System Capabilities:**

    * **🤖 Model Forge:** Train & Benchmark 6 different AI algorithms.
    * **🧠 XAI Lab:** Differential Diagnosis (Defect vs. Golden Sample).
    * **🧪 Optimization Sandbox:** Genetic Algorithm for automated repair.
    * **🏭 Digital Twin:** Real-time process simulation.

    *Select a module from the sidebar to begin.*
    """)

with col2:
    st.success("✅ **Genetic Engine:** Ready")
    st.success("✅ **Physics Constraints:** Active")
    st.success("✅ **Model Zoo:** 6 Algorithms")

st.markdown("---")
st.caption("v2.0 Enterprise Architecture | © 2026 Intelligent Systems Lab")