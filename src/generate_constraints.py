import pandas as pd
import yaml
import numpy as np
import os
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# CONFIGURATION
INPUT_FILE = "raw_data.csv"  # Updated to match your new file
# DYNAMIC PATH: Finds the 'config' folder automatically relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "config", "constraints.yaml")


def calculate_precision(series):
    """Auto-detects the sensor precision (step size)."""
    try:
        unique_vals = np.sort(series.dropna().unique())
        if len(unique_vals) < 2: return 1.0

        diffs = np.diff(unique_vals)
        min_diff = np.min(diffs[diffs > 0])

        if min_diff < 0.1: return 0.01
        if min_diff < 1.0: return 0.1
        if min_diff < 5.0: return 1.0
        return 5.0
    except:
        return 0.1


def load_data(filepath):
    if not os.path.exists(filepath):
        # Try looking in the data/raw folder if not in root
        alt_path = os.path.join(PROJECT_ROOT, "data", "raw", filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            print(f"❌ ERROR: File not found: {filepath}")
            return None

    print(f"📂 Loading data from {filepath}...")
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None


def generate_constraints():
    df = load_data(INPUT_FILE)
    if df is None: return

    constraints = {}
    print("\n⚙️  Extracting Physics Constraints...")

    for col in df.select_dtypes(include=[np.number]).columns:
        # robustly skip ID/target columns
        clean_col = col.strip().lower()
        if clean_col in ['id', 'cycle_id', 'quality', 'quality_type', 'defect_type', 'row_number', 'target']:
            continue

        min_val = float(df[col].min())
        max_val = float(df[col].max())
        step_val = calculate_precision(df[col])

        constraints[col] = {
            "min": min_val,
            "max": max_val,
            "step": step_val
        }
        print(f"   ✅ {col}: {min_val} to {max_val}")

    # Write to YAML
    yaml_content = f"""# Thesis V2 - Physics Constraints Engine
# Auto-generated from: {INPUT_FILE}
# DEFINITION:
#   These are the hard safety limits for the Injection Molding Machine.
#   The Genetic Algorithm will NEVER optimize parameters outside these bounds.

"""

    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w') as file:
        file.write(yaml_content)
        yaml.dump(constraints, file, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Success! Constraints saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_constraints()