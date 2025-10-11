from dynamic_planning import PolicyIteration, ValueIteration, print_agent

class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.P = self.createP() # 状态转移概率

    def createP(self):
        # P[state][action] = [(probability, nextstate, reward, done), ...]
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)] 
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]] # 4种动作： 0：上，1：下，2：左，3：右
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    if i == self.nrow -1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i*self.ncol + j, 0, True)]
                        continue
                    next_x = min(self.ncol -1, max(0, j + change[a][0]))
                    next_y = min(self.nrow -1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False

                    if  next_y == self.nrow -1 and next_x > 0:
                        done = True
                        if next_x != self.ncol -1:
                            reward = -100
                    
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P
    

if __name__ == "__main__":

    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])   


    agent2 = ValueIteration(env, theta, gamma)
    agent2.value_iteration()
    print_agent(agent2, action_meaning, list(range(37, 47)), [47])
