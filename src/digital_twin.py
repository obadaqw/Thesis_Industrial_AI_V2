"""
Module: digital_twin.py
Responsibility: Logic implementation for digital_twin.
Reference: Thesis Architecture Document.
"""
import pandas as pd
import numpy as np
import time
import random


class DigitalTwin:
    def __init__(self):
        print("🏭 Initializing Digital Twin Simulation...")
        # Initial State: The machine starts "Fresh"
        self.state = {
            "status": "IDLE",  # IDLE, RUNNING, STOPPED, MAINTENANCE
            "total_cycles": 0,
            "good_cycles": 0,
            "bad_cycles": 0,
            "availability": 0.98,  # Starts high (98%)
            "performance": 1.0,  # Starts perfect
            "quality": 1.0,  # Starts perfect
            "oee": 0.98  # A * P * Q
        }

    def update_metrics(self, prediction_class):
        """
        Updates the OEE metrics based on the latest AI prediction.
        Conforming classes: 1 (Acceptable) and 2 (Target) → good cycle.
        Non-conforming classes: 0 (Waste) and 3 (Inefficient) → bad cycle.
        Consistent with the ISO 9001 FPY definition and the CF acceptance threshold.
        """
        self.state["status"] = "RUNNING"
        self.state["total_cycles"] += 1

        # 1. Update Counts (Quality)
        # Conforming = {Acceptable (1), Target (2)}, consistent with the
        # ISO dashboard FPY definition and the CF acceptance threshold.
        if prediction_class in (1, 2):
            self.state["good_cycles"] += 1
        else:
            self.state["bad_cycles"] += 1

        # Recalculate Quality Ratio (Good / Total)
        if self.state["total_cycles"] > 0:
            self.state["quality"] = self.state["good_cycles"] / self.state["total_cycles"]

        # 2. Simulate Performance (Speed vs Ideal Speed)
        # We simulate minor natural fluctuations (95% - 100% efficiency)
        self.state["performance"] = random.uniform(0.95, 1.0)

        # 3. Simulate Availability (Uptime / Scheduled Time)
        # We simulate a slow degradation over time (wear and tear)
        # It drops by 0.01% every cycle unless reset
        decay = 0.0001
        self.state["availability"] = max(0.85, self.state["availability"] - decay)

        # 4. Calculate Final OEE
        # OEE = Availability * Performance * Quality
        self.state["oee"] = (
                self.state["availability"] * self.state["performance"] * self.state["quality"]
        )

        return self.state

    def reset_machine(self):
        """Resets counters (e.g., for a new shift)."""
        self.state["total_cycles"] = 0
        self.state["good_cycles"] = 0
        self.state["bad_cycles"] = 0
        self.state["quality"] = 1.0
        self.state["availability"] = 0.98
        return self.state


if __name__ == "__main__":
    # ---------------------------------------------------------
    # UNIT TEST: Verify OEE Math
    # ---------------------------------------------------------
    print("\n🏭 Testing Digital Twin Logic...")
    twin = DigitalTwin()

    # Simulate 10 Cycles
    # 6 × Target (class 2) + 1 × Acceptable (class 1) + 3 × Waste (class 0)
    # Both class 1 and class 2 are conforming → 7 good, 3 bad, quality = 70%
    outcomes = [2, 2, 2, 2, 2, 2, 1, 0, 0, 0]
    print("   Running 10 simulated cycles "
          "(6×Target, 1×Acceptable = 7 conforming; 3×Waste = 3 non-conforming)...")

    for outcome in outcomes:
        state = twin.update_metrics(outcome)

    print("\n📊 Final Machine State:")
    print(f"   Total Cycles: {state['total_cycles']}")
    print(f"   Good Cycles:  {state['good_cycles']} (Expected: 7 — class 1 and 2 both conforming)")
    print(f"   Quality:      {state['quality']:.2%} (Expected: 70%)")
    print(f"   OEE Score:    {state['oee']:.2%}")

    # Simple Assert
    if state['quality'] == 0.7:
        print("\n✅ TEST PASSED: Quality calculation is correct (Acceptable + Target both conforming).")
    else:
        print("\n❌ TEST FAILED: Math error.")