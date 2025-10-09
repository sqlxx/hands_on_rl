import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, arms):
        self.probs = np.random.uniform(size=arms)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.arms = arms

    def pull(self, arm):
        import random
        return 1 if random.random() < self.probabilities[arm] else 0

    def get_optimal_arm(self):
        return max(range(self.n_arms), key=lambda i: self.probabilities[i])