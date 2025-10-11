
import numpy as np

S = ["s1", "s2", "s3", "s4", "s5"]
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"] # 动作集合
P = {"s1-保持s1-s1": 1.0, "s1-前往s2-s2": 1.0,
     "s2-前往s1-s1": 1.0, "s2-前往s3-s3": 1.0, 
     "s3-前往s4-s4": 1.0, "s3-前往s5-s5": 1.0,
     "s4-前往s5-s5": 1.0, "s4-概率前往-s2":0.2,
     "s4-概率前往-s3":0.4, "s4-概率前往-s4":0.4}

R = {"s1-保持s1": -1, "s1-前往s2": 0,
     "s2-前往s1": -1, "s2-前往s3": -2,
     "s3-前往s4": -4, "s3-前往s5": 0,
     "s4-前往s5": 10, "s4-概率前往": 1}

Pi_1 = {
    "s1-保持s1": 0.5, "s1-前往s2": 0.5,
    "s2-前往s1": 0.5, "s2-前往s3": 0.5,
    "s3-前往s4": 0.5, "s3-前往s5": 0.5,
    "s4-前往s5": 0.5, "s4-概率前往": 0.5
}

Pi_2 = {
    "s1-保持s1": 0.6, "s1-前往s2": 0.4,
    "s2-前往s1": 0.3, "s2-前往s3": 0.7,
    "s3-前往s4": 0.5, "s3-前往s5": 0.5,
    "s4-前往s5": 0.1, "s4-概率前往": 0.9
}

def join(str1, str2):
    return str1 + "-" + str2


'''MDP: 采样函数, Pi: 策略, timestep_max: 限制最长时间步, number: 采样数量'''
def sample(MDP, Pi, timestep_max, number):
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)] # 随机选择一个初始状态
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            # 在状态s下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            # 根据状态转移概率选择下一个状态
            rand, temp = np.random.rand(), 0
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))
            s = s_next
        episodes.append(episode)
    return episodes

MDP = (S, A, P, R, 0.5)
episodes = sample(MDP, Pi_1, 20, 5)
print('第一条序列\n', episodes[0])
print('第2条序列\n', episodes[1])
print('第3条序列\n', episodes[2])
print('第5条序列\n', episodes[4])


def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for t in range(len(episode)-1, -1, -1):
            s, a, r, s_next = episode[t]
            G = gamma * G + r
            N[s] += 1
            V[s] += (G - V[s]) / N[s]
    return V

timestep_max = 50
episodes = sample(MDP, Pi_1, timestep_max, 10000)

V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}

MC(episodes, V, N, 0.5)
print("策略Pi_1下的状态值价值为:\n", V)
