import numpy as np

class BernoulliBandit:
    def __init__(self, arms):
        self.probs = np.random.uniform(size=arms)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.arms = arms

    def step(self, arm):
        if np.random.rand() < self.probs[arm]:
            return 1
        else:
            return 0   

