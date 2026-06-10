"""
Module: optimizer_genetic.py
Responsibility: Logic implementation for optimizer_genetic.
Reference: Thesis Architecture Document.
"""
import numpy as np
import pandas as pd
import joblib
import yaml
import os
import random
from copy import deepcopy

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
CONFIG_PATH = os.path.join(BASE_DIR, "config", "constraints.yaml")


class GeneticOptimizer:
    def __init__(self):
        print("🧬 Initializing Genetic Optimizer (Evolutionary Strategy)...")

        # 1. Load the Brains
        self.model = joblib.load(os.path.join(MODELS_DIR, "current_model.pkl"))
        self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        self.feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

        # 2. Load the Physics (Constraints)
        with open(CONFIG_PATH, "r") as f:
            self.constraints = yaml.safe_load(f)

        # 3. Identify Target Class (3 = Target/Optimal)
        # We find where "3" lives in the model's output [1, 2, 3, 4]
        # Model is trained with labels shifted -1 (original quality 1-4 → model class 0-3).
        # Original quality 3 (Target) = model class 2.
        try:
            self.target_class_idx = np.where(self.model.classes_ == 2)[0][0]
            print(f"   🎯 Optimization Target: Original Quality 3 → Model Class 2 (Index {self.target_class_idx})")
        except IndexError:
            print("   ⚠️ WARNING: Class 2 not found in model. Defaulting to last class.")
            self.target_class_idx = -1

    def _is_valid(self, individual):
        """Checks if a candidate solution obeys Physics Constraints."""
        for col, val in individual.items():
            if col in self.constraints:
                limits = self.constraints[col]
                if not (limits['min'] <= val <= limits['max']):
                    return False
        return True

    def _fitness(self, population_df):
        """
        Calculates fitness as P(model class 2 = original quality 3 = Target).
        """
        pop_ordered = population_df[self.feature_names]
        pop_scaled = self.scaler.transform(pop_ordered)
        probs = self.model.predict_proba(pop_scaled)

        # Return probability of the Target Class (Index for '3')
        return probs[:, self.target_class_idx]

    def optimize(self, current_defect_sample, population_size=50, generations=20, mutation_rate=0.2):
        """
        Evolves the defective parameters into Optimal ones.
        """
        print(f"   🚀 Evolving repairs for {generations} generations...")

        # STEP 1: INITIALIZATION (Create a tribe of mutants around the defect)
        population = []
        for _ in range(population_size):
            mutant = current_defect_sample.copy()
            for col in self.feature_names:
                if col in self.constraints:
                    # Randomly perturb within +/- 10% of constraint range
                    limits = self.constraints[col]
                    span = limits['max'] - limits['min']
                    change = random.uniform(-0.1, 0.1) * span
                    mutant[col] += change
                    # Clip to safety limits
                    mutant[col] = np.clip(mutant[col], limits['min'], limits['max'])
            population.append(mutant)

        pop_df = pd.DataFrame(population)

        # STEP 2: EVOLUTION LOOP
        for gen in range(generations):
            # A. Evaluate Fitness
            scores = self._fitness(pop_df)

            # Check if we found a "Perfect" solution (>95% confidence)
            best_idx = np.argmax(scores)
            if scores[best_idx] > 0.95:
                print(f"      ✨ Perfect solution found at Generation {gen}!")
                return pop_df.iloc[best_idx]

            # B. Selection (Keep top 50%)
            # Sort by score descending
            pop_df['score'] = scores
            pop_df = pop_df.sort_values('score', ascending=False).drop('score', axis=1)
            survivors = pop_df.iloc[:population_size // 2]

            # C. Crossover & Mutation (Repopulate)
            new_pop = [survivors.iloc[0].to_dict()]  # Keep the absolute best (Elitism)

            while len(new_pop) < population_size:
                # Crossover: Pick two parents
                parent1 = survivors.sample(1).iloc[0]
                parent2 = survivors.sample(1).iloc[0]
                child = parent1.copy()

                # Mix genes
                for col in self.feature_names:
                    if random.random() > 0.5:
                        child[col] = parent2[col]

                    # Mutation: Random drift
                    if random.random() < mutation_rate and col in self.constraints:
                        limits = self.constraints[col]
                        step = limits.get('step', 0.1)
                        # Mutate by small steps (Fine tuning)
                        drift = random.choice([-1, 1]) * step * random.randint(1, 5)
                        child[col] += drift
                        child[col] = np.clip(child[col], limits['min'], limits['max'])

                new_pop.append(child)

            pop_df = pd.DataFrame(new_pop)

        # STEP 3: RETURN CHAMPION
        final_scores = self._fitness(pop_df)
        best_idx = np.argmax(final_scores)
        best_solution = pop_df.iloc[best_idx]

        print(f"   🏁 Optimization Complete. Max Confidence: {final_scores[best_idx]:.2%}")
        return best_solution


if __name__ == "__main__":
    # TEST BENCH
    opt = GeneticOptimizer()

    # Create a fake "Waste" sample (using mean values)
    # In real app, this comes from the user selection
    test_sample = {}
    for col, limits in opt.constraints.items():
        test_sample[col] = (limits['min'] + limits['max']) / 2

    # Force it to be potentially bad
    test_df = pd.DataFrame([test_sample])

    print("\n🧪 Testing Optimization Loop...")
    optimized = opt.optimize(test_df.iloc[0])

    print("\n✨ Optimized Parameters (Target: Class 3):")
    print(optimized)