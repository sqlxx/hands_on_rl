import copy

class PolicyIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0]*self.env.ncol*self.env.nrow
        self.pi=[[0.25]*4 for _ in range(self.env.ncol*self.env.nrow)]
        self.theta = theta
        self.gamma = gamma 

    def policy_evaluation(self):
        cnt = 1
        while 1:
            max_diff = 0
            new_v = [0]*self.env.ncol*self.env.nrow
            for s in range(self.env.ncol*self.env.nrow):
                qsa_list = []
                for a in range(4):                        
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, s_next, r, done = res
                        qsa += p * (r + self.gamma * self.v[s_next] * (1-done))
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print("policy_evaluation iter: ", cnt)
    
    def policy_improvement(self):
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, s_next, r, done = res
                    qsa += p * (r + self.gamma * self.v[s_next] * (1-done))
                qsa_list.append(qsa)
            
            max_q = max(qsa_list)
            cntq = qsa_list.count(max_q)
            self.pi[s] = [1/cntq if q == max_q else 0 for q in qsa_list]
        
        print("policy improved")
        return self.pi

    def policy_iteration(self):
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if old_pi == new_pi: 
                break

class ValueIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow
        self.theta = theta
        self.gamma = gamma
        self.pi = [None for _ in range(self.env.ncol * self.env.nrow)]
    
    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, s_next, r, done = res
                        qsa += p * (r + self.gamma * self.v[s_next] * (1 - done))
                    qsa_list.append(qsa) # 这一行和下一行是价值迭代和策略迭代的主要区别
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print("value_iteration iter: ", cnt)
        self.get_policy()

    def get_policy(self):
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, s_next, r, done = res
                    qsa += p * (r + self.gamma * self.v[s_next] * (1 - done))
                qsa_list.append(qsa)
            max_q = max(qsa_list)
            cntq = qsa_list.count(max_q)
            self.pi[s] = [1 / cntq if q == max_q else 0 for q in qsa_list]

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值: ")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print("%6.6s" % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()
    print("策略: ")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster:
                print("****", end=' ')
            elif (i * agent.env.ncol + j) in end:
                print("EEEE", end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print() 