"""
Module: 5_🧪_Optimization_Sandbox.py
Responsibility: Logic implementation for 5_🧪_Optimization_Sandbox.
Reference: Thesis Architecture Document.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from optimizer_genetic import GeneticOptimizer
from role_manager import render_role_selector, render_access_gate

st.set_page_config(page_title="Optimization Sandbox", page_icon="🧪", layout="wide")

render_role_selector()
render_access_gate("Optimization Sandbox")

st.markdown("""
    <style>
    .metric-card { background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #444; }
    .stProgress .st-bo { background-color: #00FFCC; }
    </style>
""", unsafe_allow_html=True)

st.title("🧪 Holistic Optimization Sandbox")
st.markdown("### Automated Repair Recipe Generation (Genetic Algorithm)")


# --- INITIALIZE ENGINE ---
@st.cache_resource
def load_optimizer():
    return GeneticOptimizer()


try:
    optimizer = load_optimizer()
except Exception as e:
    st.error(f"❌ Optimizer Failed to Load: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔧 Settings")
    generations = st.slider("Generations", 10, 100, 30, help="More generations = better results but slower.")
    pop_size = st.slider("Population Size", 20, 200, 50, help="Larger population = more diverse solutions.")
    mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.2)

    st.info("""
    **How it works:**
    The Genetic Algorithm creates a 'tribe' of 50 possible machine settings. 
    It evolves them over 30 generations, keeping the ones that hit 'Class 3' (Target) 
    and killing the ones that cause defects.
    """)

# --- MAIN INTERFACE ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### 1. Input Defective Parameters")
    st.caption("Enter current machine settings to repair.")

    # Dynamic Input Fields based on constraints.yaml
    input_data = {}

    # We use a form to prevent reloading on every keystroke
    with st.form("param_input"):
        for feature, limits in optimizer.constraints.items():
            # Default to middle value
            default_val = (limits['min'] + limits['max']) / 2
            val = st.number_input(
                f"{feature}",
                min_value=float(limits['min']),
                max_value=float(limits['max']),
                value=float(default_val),
                step=float(limits['step'])
            )
            input_data[feature] = val

        submitted = st.form_submit_button("🧬 Start Evolution", type="primary")

with col2:
    if submitted:
        st.markdown("#### 🧬 Evolution Progress")

        # UI Animation elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()

        # Prepare Data
        current_sample = pd.Series(input_data)

        # Run Animation (Visual Feedback)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text(f"Generation {int(i / 3)}: Creating Initial Population...")
            elif i < 70:
                status_text.text(f"Generation {int(i / 3)}: Applying Crossover & Mutation...")
            else:
                status_text.text(f"Generation {int(i / 3)}: Selecting Fittest Survivors...")

        # --- RUN REAL OPTIMIZATION ---
        try:
            optimized_params = optimizer.optimize(
                current_sample,
                generations=generations,
                population_size=pop_size,
                mutation_rate=mutation_rate
            )

            st.success("✨ Optimization Complete! Optimal parameters found.")

            # --- RESULTS DISPLAY ---
            st.markdown("### 📋 Repair Recipe")

            # 1. Compare Table
            comparison_df = pd.DataFrame({
                'Parameter': input_data.keys(),
                'Current (Defect)': input_data.values(),
                'Optimized (Target)': [optimized_params[k] for k in input_data.keys()]
            })

            # Calculate Absolute Change
            comparison_df['Change'] = comparison_df['Optimized (Target)'] - comparison_df['Current (Defect)']


            # Highlight huge changes
            def highlight_change(val):
                color = 'green' if abs(val) > 0.0 else 'black'
                return f'color: {color}; font-weight: bold'


            st.dataframe(
                comparison_df.style.map(highlight_change, subset=['Change']),
                width='stretch'
            )

            # 2. Download Button
            csv = comparison_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Download Recipe to Machine",
                csv,
                "repair_recipe.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Optimization Error: {e}")

    else:
        st.info("👈 Enter parameters on the left and click 'Start Evolution'.")
        # Conceptual Image to fill white space
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <h3>Waiting for Input...</h3>
            <p>The system will evolve a solution that satisfies:</p>
            <ul style="display: inline-block; text-align: left;">
                <li>✅ Target Quality (Class 3)</li>
                <li>✅ Physical Constraints</li>
                <li>✅ Minimal Deviation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)