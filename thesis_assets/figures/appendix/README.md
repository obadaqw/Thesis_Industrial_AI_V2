# Appendix Screenshots — Manual Capture Checklist

These cannot be produced by a script: they require running the Streamlit
application interactively (`streamlit run app.py`) and capturing each page.
**Use the light theme** if available (Settings -> Theme), for consistency
with the print-ready figures elsewhere in `thesis_assets/` and for legible
printing/photocopying.

Capture each of the following 8 pages (see `README.md` for the role-access
gate on each):

1. **Model Forge** — AI Engineer. Show the 6-algorithm training/comparison view.
2. **XAI Lab** — AI Engineer. Show a SHAP + LIME explanation for a sample cycle.
3. **RCA Investigator** — AI Engineer / Quality Manager. Show a resolved
   3-tier counterfactual result, including the validator confirmation badge.
4. **Digital Twin** — Operator. Show the OEE gauge (availability / performance / quality).
5. **Smart Reports** — Operator / Quality Manager. Show a generated LLM shift report.
6. **ISO 9001 Dashboard** — Quality Manager. Show the Cp/Cpk process-capability view.
7. **Drift Monitor** — AI Engineer / Quality Manager. Show a PSI drift alert
   (trigger with the synthetic drift injection feature).
8. **Cycle History** — AI Engineer / Quality Manager / Operator. Show the
   SQLite audit-trail table with several logged cycles.

Save each screenshot as `fig_appendix_<page_slug>.png` in this directory
(e.g. `fig_appendix_model_forge.png`) at the highest resolution your browser
allows, then reference them from the appendix chapter. No `.pdf` vector
equivalent is required for screenshots.
