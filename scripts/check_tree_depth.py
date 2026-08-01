"""
check_tree_depth.py — Report the actual maximum, mean, and median depth
realised across the 151 estimators of the champion RandomForest.

This empirically verifies whether the configured max_depth=79 is a binding
constraint. If all trees terminate well below 79, the depth deviation
documented in docs/hyperparameter_provenance.md is immaterial in practice.

Output: models/tree_depth.json
"""

import os, sys, json
import numpy as np
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

CKPT = os.path.join(BASE, "models", "checkpoints")


def main():
    model = joblib.load(os.path.join(CKPT, "current_model.pkl"))
    depths = [est.get_depth() for est in model.estimators_]

    result = {
        "n_estimators": len(depths),
        "max_depth_configured": model.max_depth,
        "max_depth_realised":  int(max(depths)),
        "mean_depth_realised": round(float(np.mean(depths)), 2),
        "median_depth_realised": round(float(np.median(depths)), 2),
        "min_depth_realised":  int(min(depths)),
        "binding": int(max(depths)) >= model.max_depth,
    }

    out = os.path.join(BASE, "models", "tree_depth.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Configured max_depth : {model.max_depth}")
    print(f"Realised max depth   : {result['max_depth_realised']}")
    print(f"Realised mean depth  : {result['mean_depth_realised']}")
    print(f"Realised median depth: {result['median_depth_realised']}")
    print(f"Realised min depth   : {result['min_depth_realised']}")
    print(f"Constraint binding?  : {result['binding']}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
