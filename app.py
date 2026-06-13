import streamlit as st
import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── First-boot initializer ────────────────────────────────────────────────────
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
    _seed_demo_cycles()
    print("✅  First boot complete.", flush=True)


def _seed_demo_cycles():
    """Populate cycle history with realistic demo data on first boot."""
    try:
        import cycle_store, random
        if cycle_store.get_stats()["total"] > 0:
            return
        random.seed(42)
        scenarios = [
            # (prediction, rca_tier, rca_status, validator_ok_prob)
            (2, 0, "already_acceptable", 1.0),   # Target — no repair needed
            (2, 0, "already_acceptable", 1.0),
            (1, 0, "already_acceptable", 1.0),   # Acceptable
            (0, 1, "resolved",           0.9),   # Waste → Tier 1 fix
            (0, 1, "resolved",           0.9),
            (0, 2, "resolved",           0.7),   # Waste → Tier 2 fix
            (3, 1, "resolved",           0.85),  # Inefficient → Tier 1
            (0, 3, "escalate",           0.0),   # Escalation
        ]
        weights = [25, 20, 15, 12, 10, 8, 7, 3]
        for i in range(60):
            s = random.choices(scenarios, weights=weights)[0]
            pred, tier, status, vok_p = s
            conf = round(random.uniform(0.62, 0.98), 3)
            cf_conf = round(random.uniform(0.56, 0.92), 3) if tier > 0 else 0.0
            vok = random.random() < vok_p if tier > 0 else False
            cycle_store.log_cycle(
                cycle_id=i, prediction=pred, confidence=conf,
                rca_tier=tier, rca_status=status,
                cf_confidence=cf_conf, validator_ok=vok
            )
        print("📋  Demo cycle history seeded (60 cycles).", flush=True)
    except Exception as e:
        print(f"   ⚠️ Seed skipped: {e}", flush=True)


_bootstrap()

# Start background API server
try:
    from platform_server import start_background
    start_background(port=8502)
except Exception:
    pass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Shadow V2",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css():
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp { background-color: #0E1117; color: #FAFAFA; }
            .stSidebar { background-color: #262730; }
            h1, h2, h3 { color: #00FFCC !important; }
            .stButton>button { border-radius: 5px; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)


load_css()

# Sidebar
try:
    from role_manager import render_role_selector
    render_role_selector()
except Exception:
    pass

st.sidebar.title("🏭 Digital Shadow V2")
st.sidebar.markdown("---")
st.sidebar.success("**System:** Online 🟢")
st.sidebar.caption("API: http://localhost:8502/docs")

# ── Load live metrics (safe — all wrapped in try/except) ──────────────────────
import numpy as np
import pandas as pd

@st.cache_data(ttl=60)
def _load_metrics():
    import joblib
    from sklearn.metrics import accuracy_score
    root = os.path.dirname(os.path.abspath(__file__))
    ckpt = os.path.join(root, "models", "checkpoints")
    proc = os.path.join(root, "data", "processed")

    model  = joblib.load(os.path.join(ckpt, "current_model.pkl"))
    X_val  = pd.read_csv(os.path.join(proc, "X_val.csv"))
    y_val  = pd.read_csv(os.path.join(proc, "y_val.csv")).values.ravel()
    y_val  = y_val - 1 if y_val.min() > 0 else y_val
    y_pred = model.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)

    probas     = model.predict_proba(X_val)
    good_mask  = (y_pred == 1) | (y_pred == 2)
    oee_proxy  = good_mask.mean()

    return {
        "algo":    type(model).__name__,
        "acc":     acc,
        "n_val":   len(y_val),
        "oee":     oee_proxy,
        "classes": int(len(model.classes_)),
    }


@st.cache_data(ttl=30)
def _load_psi_status():
    import joblib
    from drift_detector import DriftDetector, PSI_CRITICAL, PSI_MODERATE
    root = os.path.dirname(os.path.abspath(__file__))
    ckpt = os.path.join(root, "models", "checkpoints")
    proc = os.path.join(root, "data", "processed")
    feat  = joblib.load(os.path.join(ckpt, "feature_names.pkl"))
    X_tr  = pd.read_csv(os.path.join(proc, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(proc, "X_val.csv"))
    det   = DriftDetector(X_tr, feat)
    psi   = det.compute_all_psi(X_val)
    return det.overall_status(psi), int((psi["PSI"] >= PSI_CRITICAL).sum())


metrics = None
try:
    metrics = _load_metrics()
except Exception:
    pass

psi_status, n_critical = "—", 0
try:
    psi_status, n_critical = _load_psi_status()
except Exception:
    pass

try:
    import cycle_store as _cs
    ch_stats = _cs.get_stats()
    recent   = _cs.get_recent(n=5)
except Exception:
    ch_stats = {"total": 0, "good": 0, "fpy": 0.0, "tier_counts": {}}
    recent   = pd.DataFrame()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏭 Intelligent Quality Control Center")
st.markdown("##### Master's Thesis — Explainable AI for Injection Molding | XAI · RCA · ISO 9001 · Digital Twin")
st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

if metrics:
    k1.metric("Model", metrics["algo"])
    k2.metric("Val Accuracy", f"{metrics['acc']:.2%}")
    k3.metric("OEE (proxy)", f"{metrics['oee']:.1%}")
else:
    k1.metric("Model", "Loading…")
    k2.metric("Val Accuracy", "—")
    k3.metric("OEE (proxy)", "—")

_psi_icon = {"stable": "🟢 Stable", "moderate": "🟡 Moderate",
             "critical": "🔴 Critical"}.get(psi_status, "—")
k4.metric("Drift Status", _psi_icon)
k5.metric("Cycles Logged", ch_stats["total"])

st.markdown("---")

# ── Main columns ──────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.markdown("#### 🧩 System Modules")

    MODULE_STATUS = [
        ("🤖", "Model Forge",          "Train & benchmark 6 ML algorithms",             "✅"),
        ("🧠", "XAI Lab",              "SHAP + LIME differential diagnosis",             "✅"),
        ("🕵️", "RCA Investigator",    "3-tier counterfactual root-cause analysis",      "✅"),
        ("🏭", "Digital Twin",         "OEE simulation & live cycle monitoring",         "✅"),
        ("🧪", "Optimization Sandbox", "Genetic algorithm repair recipe generation",     "✅"),
        ("📑", "Smart Reports",        "Llama 3.3 / Groq automated shift reports",       "✅"),
        ("📊", "ISO 9001 Dashboard",   "Cp/Cpk · SPC · non-conformance register",        "✅"),
        ("📡", "Drift Monitor",        "PSI + Page-Hinkley concept drift detection",     "✅"),
        ("📋", "Cycle History",        "SQLite audit trail — ISO 9001 traceability",     "✅"),
    ]

    for icon, name, desc, status in MODULE_STATUS:
        st.markdown(
            f"<div style='display:flex; align-items:center; padding:6px 0; "
            f"border-bottom:1px solid #2a2a2a;'>"
            f"<span style='font-size:1.2em; width:32px'>{icon}</span>"
            f"<span style='flex:1'><b>{name}</b> — <span style='color:#888;font-size:0.9em'>{desc}</span></span>"
            f"<span style='color:#00CC88; font-weight:bold'>{status}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

with right:
    st.markdown("#### 📋 Recent Cycle Activity")
    if recent.empty:
        st.info("No cycles logged yet. Run an RCA analysis to populate.")
    else:
        CLASS_NAMES = {0: "Waste ❌", 1: "Acceptable", 2: "Target ✅", 3: "Inefficient ⚠️"}
        TIER_SHORT  = {0: "OK", 1: "T1", 2: "T2", 3: "T3 🚨"}
        disp = recent[["timestamp", "cycle_id", "prediction", "rca_tier", "confidence"]].copy()
        disp["Quality"] = disp["prediction"].map(CLASS_NAMES)
        disp["Tier"]    = disp["rca_tier"].map(TIER_SHORT)
        disp["Conf"]    = disp["confidence"].round(2)
        st.dataframe(
            disp[["timestamp", "cycle_id", "Quality", "Tier", "Conf"]],
            width='stretch', hide_index=True
        )

    st.markdown("---")
    st.markdown("#### 📈 FPY Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cycles", ch_stats["total"])
    c2.metric("First Pass Yield", f"{ch_stats['fpy']:.1%}" if ch_stats["total"] else "—")
    c3.metric("Tier 3 Escalations", ch_stats["tier_counts"].get(3, 0))

st.markdown("---")

# ── Tech stack footer ─────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.8em; padding:8px 0'>"
    "Random Forest · SHAP · LIME · Genetic Algorithm · Groq/Llama-3.3 · "
    "FastAPI · SQLite · Telegram · Streamlit · Docker"
    "<br>v2.0 — Master's Thesis 2026"
    "</div>",
    unsafe_allow_html=True
)
