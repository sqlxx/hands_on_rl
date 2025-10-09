import numpy as np

class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.arms)
        self.regret = 0
        self.actions =  []
        self.regrets = []
    
    def update_regres(self, arm):
        self.regret += self.bandit.best_prob - self.bandit.probs[arm]
        self.regrets.append(self.regret)
    
    def run_one_step(self):
        raise NotImplementedError
    
    def run(self, num_steps):
        for _ in range(num_steps):
            arm = self.run_one_step()
            self.actions.append(arm)
            self.counts[arm] += 1
            self.update_regres(arm)
    

class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super().__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob]*self.bandit.arms)
    
    def run_one_step(self):
        val = np.random.rand()
        if val < self.epsilon:
            arm = np.random.randint(0, self.bandit.arms)
        else:
            arm = np.argmax(self.estimates)
        
        r = self.bandit.step(arm)
        self.estimates[arm] += 1./(self.counts[arm] + 1) * (r - self.estimates[arm])
        
        return arm    

class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super().__init__(bandit)
        self.estimates = np.array([init_prob]*self.bandit.arms)
        self.total_counts = 0
    
    def run_one_step(self):
        self.total_counts += 1
        if np.random.rand() < 1/self.total_counts:
            arm = np.random.randint(0, self.bandit.arms)
        else:
            arm = np.argmax(self.estimates)
        
        r = self.bandit.step(arm)
        self.estimates[arm] += 1./(self.counts[arm] + 1) * (r - self.estimates[arm])
        return arm

class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super().__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob]*self.bandit.arms)
        self.coef = coef
    
    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count)/(2*(self.counts +1))) # 计算上置信界
        arm = np.argmax(ucb)
        
        r = self.bandit.step(arm)
        self.estimates[arm] += 1./(self.counts[arm] + 1) * (r - self.estimates[arm])
        return arm

class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super().__init__(bandit)
        self.alphas = np.ones(self.bandit.arms)
        self.betas = np.ones(self.bandit.arms)
    
    def run_one_step(self):
        samples = np.random.beta(self.alphas, self.betas)
        arm = np.argmax(samples)
        
        r = self.bandit.step(arm)
        if r == 1:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1
        return arm