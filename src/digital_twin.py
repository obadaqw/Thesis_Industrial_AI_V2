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
        Input: prediction_class (3 = Target/Good, others = Bad)
        """
        self.state["status"] = "RUNNING"
        self.state["total_cycles"] += 1

        # 1. Update Counts (Quality)
        # Model class 2 = original quality 3 (Target). Class 3 = Inefficient.
        if prediction_class == 2:
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
    # We will force 7 Good cycles (Class 2 = original quality 3) and 3 Bad cycles (Class 0)
    print("   Running 10 simulated cycles (7 Good, 3 Bad)...")

    for i in range(10):
        if i < 7:
            outcome = 2  # Good (original quality 3 = Target)
        else:
            outcome = 0  # Bad (original quality 1 = Waste)

        state = twin.update_metrics(outcome)

    print("\n📊 Final Machine State:")
    print(f"   Total Cycles: {state['total_cycles']}")
    print(f"   Good Cycles:  {state['good_cycles']} (Expected: 7)")
    print(f"   Quality:      {state['quality']:.2%} (Expected: 70%)")
    print(f"   OEE Score:    {state['oee']:.2%}")

    # Simple Assert
    if state['quality'] == 0.7:
        print("\n✅ TEST PASSED: Quality calculation is correct.")
    else:
        print("\n❌ TEST FAILED: Math error.")