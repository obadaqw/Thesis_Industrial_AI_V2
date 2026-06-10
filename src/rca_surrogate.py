import pandas as pd
import numpy as np
import joblib
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "checkpoints")


class RCASurrogate:
    """
    Rule-based anomaly detector using z-scores computed from raw training
    statistics (mean/std saved in train_stats.pkl).  Decoupled from the
    MinMaxScaler so z-scores remain statistically meaningful.
    """

    # Maps every sensor feature to a physical machine component.
    COMPONENT_MAP = {
        "Melt temperature":                             "Heating Unit (Zone 1-3)",
        "Mold temperature":                             "Cooling Circuit / Chiller",
        "time_to_fill":                                 "Flow Controller / Nozzle",
        "ZDx - Plasticizing time":                      "Plasticizing Unit / Screw",
        "ZUx - Cycle time":                             "PLC Timer / Robot Arm",
        "SKx - Closing force":                          "Clamping Unit",
        "SKs - Clamping force peak value":              "Clamping Unit",
        "Ms - Torque peak value current cycle":         "Screw Drive Motor",
        "Mm - Torque mean value current cycle":         "Screw Drive Motor",
        "APSs - Specific back pressure peak value":     "Back Pressure Valve",
        "APVs - Specific injection pressure peak value":"Injection Unit / Nozzle",
        "CPn - Screw position at the end of hold pressure": "Hold Pressure Controller",
        "SVo - Shot volume":                            "Shot Volume Controller",
    }

    def __init__(self):
        print("🕵️ Initializing RCA Surrogate...")
        try:
            self.feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
            # Use training-set mean/std for z-score computation.
            # This is independent of the scaler type (MinMaxScaler / StandardScaler).
            stats = joblib.load(os.path.join(MODELS_DIR, "train_stats.pkl"))
            self.train_mean = stats['mean'].values
            self.train_std  = stats['std'].values
            print("✅ RCA Surrogate Ready.")
        except Exception as e:
            print(f"❌ RCA Init Failed: {e}")
            self.feature_names = None

    def analyze_cycle(self, cycle_data_real):
        """
        Input:  DataFrame with real (unscaled) physics values.
        Output: dict with critical_anomalies, warnings, system_health_score.
        """
        if self.feature_names is None:
            return {"status": "Error", "report": "System Offline"}

        try:
            cycle_ordered = cycle_data_real[self.feature_names]
        except KeyError as e:
            missing = set(self.feature_names) - set(cycle_data_real.columns)
            return {"status": "Error", "report": f"Missing sensor data: {missing}"}

        # Z-score = (value - training_mean) / training_std
        values = cycle_ordered.values[0]
        z_scores = (values - self.train_mean) / (self.train_std + 1e-9)

        diagnosis = {
            "critical_anomalies": [],
            "warnings": [],
            "system_health_score": 100.0
        }

        for idx, z in enumerate(z_scores):
            feature  = self.feature_names[idx]
            abs_z    = abs(z)
            component = self.COMPONENT_MAP.get(feature, "Sensor System")

            if abs_z > 3.0:
                diagnosis["critical_anomalies"].append({
                    "sensor":    feature,
                    "component": component,
                    "value":     round(float(values[idx]), 4),
                    "deviation": f"{z:.1f}σ",
                    "action":    "IMMEDIATE INSPECTION REQUIRED"
                })
                diagnosis["system_health_score"] -= 20

            elif abs_z > 2.0:
                diagnosis["warnings"].append({
                    "sensor":    feature,
                    "component": component,
                    "value":     round(float(values[idx]), 4),
                    "deviation": f"{z:.1f}σ",
                    "action":    "Monitor for drift"
                })
                diagnosis["system_health_score"] -= 5

        diagnosis["system_health_score"] = max(0.0, min(100.0, diagnosis["system_health_score"]))
        return diagnosis


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    rca = RCASurrogate()
    if rca.feature_names:
        import pandas as pd
        stats = joblib.load(os.path.join(MODELS_DIR, "train_stats.pkl"))
        # Test with mean values (should be healthy)
        mean_row = pd.DataFrame([stats['mean'].values], columns=rca.feature_names)
        normal = rca.analyze_cycle(mean_row)
        print("Normal cycle health:", normal['system_health_score'])
        # Force a critical deviation (+5 sigma on first feature)
        fault_row = mean_row.copy()
        fault_row.iloc[0, 0] += stats['std'].iloc[0] * 5
        fault = rca.analyze_cycle(fault_row)
        print("Fault cycle health:", fault['system_health_score'],
              "| critical anomalies:", len(fault['critical_anomalies']))
