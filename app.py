import streamlit as st
import os

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

# 3. SIDEBAR HEADER
st.sidebar.image("https://img.icons8.com/color/96/000000/industrial-robot.png", width=80)
st.sidebar.title("🏭 Digital Shadow")
st.sidebar.markdown("---")
st.sidebar.info("**Status:** System Online 🟢")

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