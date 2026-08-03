"""
export_quality_modules.py — Quality 4.0 module exports for thesis Chapter 4/5.

Calls the existing iso9001_metrics, digital_twin, cycle_store and llm_wrapper
modules against real data; reimplements none of their logic.

Outputs (thesis_assets/data/):
  process_capability.json   Cp/Cpk per feature (iso9001_metrics.compute_all_capabilities)
  oee_simulation.json       one representative OEE scenario (DigitalTwin)
  traceability_sample.json  5 demonstration audit-trail records (cycle_store schema)
  llm_report_sample.md      one verbatim Groq shift-report exhibit (skipped if no API key)

Usage:
    python scripts/export_quality_modules.py

Note on process_capability.json: config/constraints.yaml states in its own
header that it was "Auto-generated from: raw_data.csv" — i.e. the USL/LSL
values are the OBSERVED min/max of the historical dataset, not sourced
customer or engineering specification limits. This script computes Cp/Cpk
against those bounds and labels them explicitly as ASSUMED, per the task
requirement not to present assumed limits as sourced ones.

Note on traceability_sample.json: no interactive session has logged cycles to
data/cycle_history.db yet (the file does not exist at the time this script
was written). Rather than fabricate audit-trail values, this script exercises
the real cycle_store.log_cycle()/get_recent() functions against five genuine,
already-computed RCA outcomes from models/rca_results_test.csv, writing to a
throwaway SQLite file in the OS temp directory (never the production
data/cycle_history.db). The exported JSON states this provenance explicitly.
"""

import os
import sys
import json
import random
import tempfile
import warnings

import pandas as pd
import numpy as np
import joblib
import yaml

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

from iso9001_metrics import compute_all_capabilities  # noqa: E402
from digital_twin import DigitalTwin  # noqa: E402
import cycle_store  # noqa: E402
from rca_surrogate import RCASurrogate  # noqa: E402
from llm_wrapper import LLMWrapper  # noqa: E402

CKPT = os.path.join(BASE, "models", "checkpoints")
PROC = os.path.join(BASE, "data", "processed")
RAW_PATH = os.path.join(BASE, "data", "raw", "raw_data.csv")
CONSTRAINTS_PATH = os.path.join(BASE, "config", "constraints.yaml")
RCA_RESULTS_TEST = os.path.join(BASE, "models", "rca_results_test.csv")
CF_CASES_JSON = os.path.join(BASE, "thesis_assets", "data", "counterfactual_cases.json")
OUT_DIR = os.path.join(BASE, "thesis_assets", "data")

CLASS_NAMES = {0: "Waste", 1: "Acceptable", 2: "Target", 3: "Inefficient"}


def export_process_capability(feature_names: list) -> dict:
    if not os.path.exists(CONSTRAINTS_PATH):
        raise FileNotFoundError(f"Required source artifact missing: {CONSTRAINTS_PATH}")
    with open(CONSTRAINTS_PATH) as f:
        constraints = yaml.safe_load(f)

    df_raw = pd.read_csv(RAW_PATH)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    X_real = df_raw[feature_names]

    cap_df = compute_all_capabilities(X_real, constraints)

    result = {
        "spec_limit_source": "ASSUMED",
        "spec_limit_provenance": (
            "config/constraints.yaml, which its own header documents as "
            "'Auto-generated from: raw_data.csv' -- these are the OBSERVED "
            "min/max of the full historical dataset, not customer or "
            "engineering specification limits. Cp/Cpk below are computed "
            "against these assumed bounds."
        ),
        "population": "full raw dataset (n=%d), matching the population the "
                       "bounds were themselves derived from" % len(df_raw),
        "capabilities": cap_df.to_dict("records"),
    }
    with open(os.path.join(OUT_DIR, "process_capability.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def export_oee_simulation(model, X_test: pd.DataFrame, seed: int = 42) -> dict:
    random.seed(seed)
    twin = DigitalTwin()
    y_pred = model.predict(X_test.values)

    history = []
    for i, pred in enumerate(y_pred):
        state = twin.update_metrics(int(pred))
        if (i + 1) % 50 == 0 or (i + 1) == len(y_pred):
            history.append({"cycle": i + 1, **{k: round(v, 4) if isinstance(v, float) else v
                                                for k, v in state.items()}})

    final_state = twin.state
    result = {
        "scenario_definition": (
            f"{len(y_pred)} test-split samples replayed through DigitalTwin in "
            "index order, using the champion model's own predictions (not "
            "ground-truth labels) as the per-cycle outcome -- i.e. what the "
            "production line would have shown the operator in real time."
        ),
        "random_seed": seed,
        "stochastic_note": (
            "DigitalTwin.update_metrics() draws performance from "
            "random.uniform(0.95, 1.0) per cycle and decays availability "
            "deterministically; the seed above makes THIS run reproducible, "
            "but the module itself is inherently stochastic by design."
        ),
        "final_state": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in final_state.items()},
        "history_every_50_cycles": history,
    }
    with open(os.path.join(OUT_DIR, "oee_simulation.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def export_traceability_sample() -> dict:
    if not os.path.exists(RCA_RESULTS_TEST):
        raise FileNotFoundError(f"Required source artifact missing: {RCA_RESULTS_TEST}")
    df = pd.read_csv(RCA_RESULTS_TEST)

    # Pick 5 rows spanning the distinct (tier, status, validator_ok) outcome
    # combinations present in the test-split results, most-diverse first.
    df["_combo"] = list(zip(df["tier"], df["status"], df["validator_ok"]))
    picked = df.drop_duplicates("_combo").head(5)
    if len(picked) < 5:
        picked = pd.concat([picked, df.drop(picked.index).head(5 - len(picked))])

    import uuid
    # Deliberately NOT tempfile.gettempdir(): on this machine it resolves to
    # C:\Users\...\AppData\Local\Temp, and the C: drive has been observed
    # completely full (0 bytes free) -- SQLite fails with "database or disk
    # is full" there even though this repo's own drive has ample space.
    # Use a scratch dir on the same drive as the repo instead.
    tmp_dir = os.path.join(BASE, ".thesis_assets_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_db = os.path.join(tmp_dir, f"demo_cycle_history_{uuid.uuid4().hex}.db")
    original_db_path = cycle_store.DB_PATH
    cycle_store.DB_PATH = tmp_db
    try:
        for _, row in picked.iterrows():
            cycle_store.log_cycle(
                cycle_id=int(row["sample_idx"]),
                prediction=int(row["pred_class"]),
                confidence=float(row["input_conf"]),
                rca_tier=int(row["tier"]),
                rca_status=str(row["status"]),
                cf_confidence=float(row["cf_confidence"]),
                validator_ok=bool(row["validator_ok"]),
            )
        records_df = cycle_store.get_recent(5)
    finally:
        cycle_store.DB_PATH = original_db_path
        try:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)
            os.rmdir(tmp_dir)  # only succeeds if empty
        except OSError:
            pass  # best-effort cleanup of a throwaway scratch file/dir

    records = records_df.drop(columns=["id"]).to_dict("records")
    result = {
        "provenance": (
            "data/cycle_history.db had no logged cycles at export time (no "
            "interactive Streamlit session has run yet). These 5 records were "
            "produced by replaying genuine, already-computed RCA outcomes from "
            "models/rca_results_test.csv through the real cycle_store.log_cycle() "
            "/ get_recent() functions against a throwaway SQLite file -- the "
            "production database was not written to. No fields required "
            "anonymisation: the cycle_store schema records only "
            "timestamp/prediction/RCA-outcome fields, no operator or personal data."
        ),
        "schema_source": "src/cycle_store.py",
        "records": records,
    }
    with open(os.path.join(OUT_DIR, "traceability_sample.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def export_llm_report_sample(feature_names: list) -> str:
    out_path = os.path.join(OUT_DIR, "llm_report_sample.md")
    llm = LLMWrapper()
    if not llm.use_real_ai:
        with open(out_path, "w") as f:
            f.write("# LLM Shift Report Exhibit -- SKIPPED\n\n"
                    "GROQ_API_KEY was not available at export time, so no real "
                    "shift report could be generated. Per task instructions, "
                    "no report was fabricated.\n")
        print("  GROQ_API_KEY unavailable -- skipped, not fabricated.")
        return "skipped"

    scaler = joblib.load(os.path.join(CKPT, "scaler.pkl"))
    X_test = pd.read_csv(os.path.join(PROC, "X_test.csv"))
    surrogate = RCASurrogate()

    # Use the same sample as the Task 3 validator-confirmed worked case, so
    # the RCA surrogate diagnosis, digital-twin OEE, and LLM narrative in this
    # exhibit are all about one consistent, already-documented cycle.
    sample_idx = 39
    cf_result = None
    if os.path.exists(CF_CASES_JSON):
        with open(CF_CASES_JSON) as f:
            cf_cases = json.load(f)
        confirmed = cf_cases["cases"]["validator_confirmed"]
        if confirmed["sample_index_in_test_split"] == sample_idx:
            cf_result = {
                "status": confirmed["status"],
                "tier": confirmed["tier"],
                "adjustments": confirmed["counterfactual"]["adjustments"],
                "validator_ok": confirmed["validator_ok"],
            }
    else:
        print("  Note: thesis_assets/data/counterfactual_cases.json not found "
              "(run export_rca_cases.py first for a richer exhibit); "
              "proceeding without a counterfactual section.")

    row_real = pd.DataFrame(
        scaler.inverse_transform(X_test.iloc[[sample_idx]]), columns=feature_names
    )
    diagnosis = surrogate.analyze_cycle(row_real)

    random.seed(42)
    twin = DigitalTwin()
    for _ in range(10):
        twin.update_metrics(random.choice([0, 1, 2, 3]))
    oee_stats = {"oee": twin.state["oee"]}

    report_text = llm.generate_smart_report(diagnosis, oee_stats, cf_result=cf_result)

    # generate_smart_report silently falls back to a canned "OFFLINE REPORT"
    # string (see llm_wrapper._mock_response) on ANY Groq failure, including
    # an expired/invalid key -- which is exactly what happened at export
    # time (401 invalid_request_error). A mock string is not a real LLM
    # exhibit, so treat this the same as "API key unavailable": skip, do not
    # present the fallback text as a genuine generated report.
    if report_text.strip().startswith("**OFFLINE REPORT**"):
        with open(out_path, "w") as f:
            f.write("# LLM Shift Report Exhibit -- SKIPPED\n\n"
                    "A GROQ_API_KEY was configured but the API call failed at "
                    "export time (observed: HTTP 401, expired_api_key). "
                    "llm_wrapper.py silently falls back to a templated mock "
                    "string in this situation; per task instructions that "
                    "mock text is not presented here as a genuine exhibit. "
                    "Refresh the key and re-run scripts/export_quality_modules.py "
                    "to produce a real exhibit.\n")
        print("  GROQ_API_KEY rejected by the API (expired) -- skipped, "
              "mock fallback not presented as a genuine exhibit.")
        return "skipped"

    with open(out_path, "w") as f:
        f.write("# LLM Shift Report Exhibit\n\n")
        f.write(f"Generated from test-split sample index {sample_idx}. "
                f"System health score: {diagnosis['system_health_score']}%. "
                f"OEE (illustrative): {oee_stats['oee']:.2%}.\n\n")
        f.write("---\n\n")
        f.write(report_text)
        f.write("\n")
    print(f"  Report generated ({len(report_text)} chars).")
    return report_text


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Quality 4.0 module export ===")

    feature_names = joblib.load(os.path.join(CKPT, "feature_names.pkl"))
    model = joblib.load(os.path.join(CKPT, "current_model.pkl"))
    X_test = pd.read_csv(os.path.join(PROC, "X_test.csv"))

    print("\n[1/4] Process capability (Cp/Cpk)...")
    cap = export_process_capability(feature_names)
    n_not_capable = sum(1 for r in cap["capabilities"] if r["Status"] == "Not Capable")
    print(f"  {n_not_capable}/{len(cap['capabilities'])} features 'Not Capable' "
          f"against assumed (observed-range) limits.")

    print("\n[2/4] OEE simulation...")
    oee = export_oee_simulation(model, X_test)
    print(f"  Final OEE: {oee['final_state']['oee']:.2%}")

    print("\n[3/4] Traceability sample...")
    export_traceability_sample()

    print("\n[4/4] LLM shift-report exhibit...")
    export_llm_report_sample(feature_names)

    print(f"\nDone. Files written to {OUT_DIR}")


if __name__ == "__main__":
    main()
