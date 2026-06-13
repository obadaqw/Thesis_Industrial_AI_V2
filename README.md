# Intelligent Quality Control Center
### Master's Thesis — Explainable AI for Injection Molding

A production-grade Streamlit application that applies Explainable AI (XAI) to injection molding quality control. Trains 6 ML models, runs 3-tier counterfactual root-cause analysis, monitors process drift, and generates LLM-powered shift reports — all backed by a role-based access system and ISO 9001 compliance dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Streamlit UI (8 pages)                │
│  Model Forge · XAI Lab · RCA Investigator · Digital Twin    │
│  Smart Reports · ISO 9001 Dashboard                         │
│  Drift Monitor · Cycle History                              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      src/ (core modules)                    │
│                                                             │
│  data_pipeline.py      MinMaxScaler(-1,1), train/val split  │
│  model_factory.py      6-algorithm zoo + 5-fold CV          │
│  xai_engine.py         SHAP TreeExplainer + LIME (cached)  │
│  counterfactual_rca.py 3-tier CF engine + MLP validator     │
│  rca_surrogate.py      Z-score component health scoring     │
│  digital_twin.py       OEE simulation (A × P × Q)          │
│  drift_detector.py     PSI + Page-Hinkley concept drift     │
│  iso9001_metrics.py    Cp/Cpk process capability            │
│  llm_wrapper.py        Groq/Llama-3.3-70b shift reports    │
│  telegram_notifier.py  Async push alerts                    │
│  cycle_store.py        SQLite audit trail                   │
│  experiment_tracker.py JSON experiment history              │
│  role_manager.py       3-role access gate                   │
│  config.py             Unified secret resolution            │
└─────────────────────────────────────────────────────────────┘
```

### ML Model
- **Champion:** Random Forest — `n_estimators=151, max_depth=79, criterion='entropy'`
- **Classes:** 0=Waste · 1=Acceptable · 2=Target · 3=Inefficient
- **Scaler:** MinMaxScaler fitted on training data only (range −1 to 1)
- **Validation accuracy:** ≥ 90%

### 3-Tier Counterfactual RCA
| Tier | Method | Threshold |
|------|--------|-----------|
| 0 | Already acceptable | P(Acc)+P(Target) ≥ 0.55 |
| 1 | SHAP selects top-5 · LIME gives direction | Same |
| 2 | 15 nearest-neighbor centroid (top-7 features) | Same |
| 3 | Escalation — structurally uncorrectable | — |

An independent MLP validator (seed=7) confirms every accepted counterfactual.

---

## Quick Start

### Local (with .env)

```bash
# 1. Clone
git clone https://github.com/obadaqw/Thesis_Industrial_AI_V2.git
cd Thesis_Industrial_AI_V2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env (never commit this)
cp .env.example .env   # fill in your keys

# 4. Run data pipeline + train champion model
python src/data_pipeline.py
python src/model_factory.py --algo RF

# 5. Launch
streamlit run app.py
```

Open http://localhost:8501

### Docker

```bash
docker compose up --build
```

App at http://localhost:8501

### Streamlit Cloud

1. Add secrets in the Streamlit Cloud dashboard (Settings → Secrets):
   ```toml
   GROQ_API_KEY = "..."
   TELEGRAM_BOT_TOKEN = "..."
   TELEGRAM_CHAT_ID = "..."
   ```
2. Or copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` locally.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes (for LLM reports) | Groq Cloud API key |
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | No | Telegram chat/user ID |

---

## Pages

| # | Page | Role |
|---|------|------|
| 1 | 🤖 Model Forge | AI Engineer |
| 2 | 🧠 XAI Lab | AI Engineer |
| 3 | 🕵️ RCA Investigator | AI Engineer · Quality Manager |
| 4 | 🏭 Digital Twin | Operator |
| 5 | 📑 Smart Reports | Operator · Quality Manager |
| 6 | 📊 ISO 9001 Dashboard | Quality Manager |
| 7 | 📡 Drift Monitor | AI Engineer · Quality Manager |
| 8 | 📋 Cycle History | AI Engineer · Quality Manager · Operator |

---

## Testing

```bash
# Unit tests only (fast, ~6 s)
pytest -m unit

# Integration tests (loads models, ~14 s)
pytest -m integration

# Full suite with coverage
pytest
```

---

## Project Structure

```
Thesis_Industrial_AI_V2/
├── app.py                    # Entry point
├── pages/                    # Streamlit multi-page app (8 pages)
├── src/                      # Core ML + business logic
├── data/
│   ├── raw/                  # Original dataset
│   ├── processed/            # Train/val splits (gitignored)
│   └── golden_store/         # Golden reference samples
├── models/
│   └── checkpoints/          # Trained models + scaler (gitignored)
├── tests/                    # pytest suite (unit + integration)
├── assets/                   # CSS theme
├── constraints.yaml          # Physical sensor bounds (USL/LSL)
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml  # GitHub Actions CI
```
