import numpy as np
from matplotlib import pyplot as plt

from bernoulli_bandit import BernoulliBandit
from solvers import EpsilonGreedy, DecayingEpsilonGreedy, UCB, ThompsonSampling

def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label = solver_names[idx])
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title('%d-armed Bernoulli Bandit' % solvers[0].bandit.arms)
    plt.legend()
    plt.show()

def epsilon_solver(bandit):
    epsilons = [1e-3, 0.01, 0.1, 0.25, 0.3]
    epsilon_greedy_solvers = [EpsilonGreedy(bandit, epsilon=eps) for eps in epsilons]
    epsilon_greedy_solver_names = ['Epsilon=%.3f' % eps for eps in epsilons]
    for solver in epsilon_greedy_solvers:
        solver.run(5000)
        print("%.4f Epsilon Greedy cumulative regret: %.2f" %(solver.epsilon, solver.regret))

    plot_results(epsilon_greedy_solvers, epsilon_greedy_solver_names)

def declaying_epsilon_solver(bandit):
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit)
    decaying_epsilon_greedy_solver.run(5000)
    print("Decaying Epsilon Greedy cumulative regret: %.2f" % decaying_epsilon_greedy_solver.regret)
    plot_results([decaying_epsilon_greedy_solver], ['Decaying Epsilon Greedy'])

def ucb_solver(bandit):
    coef = 1
    ucb_solver = UCB(bandit, coef)
    ucb_solver.run(5000)
    print("UCB cumulative regret: %.2f" % ucb_solver.regret)
    plot_results([ucb_solver], ['UCB'])

def thompson_sampling(bandit):
    thompson_sampling_solver = ThompsonSampling(bandit)
    thompson_sampling_solver.run(5000)
    print("Thompson Sampling cumulative regret: %.2f" % thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver], ['Thompson Sampling'])

if __name__ == "__main__":
    np.random.seed(1) # for reproducibility
    arms = 10
    bandit_10_arm = BernoulliBandit(arms)
    print("%d-armed bandit instance" % arms)
    print("Best arm: %d, probability: %.4f" % (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

    # epsilon_solver(bandit_10_arm)
    # declaying_epsilon_solver(bandit_10_arm)
    # ucb_solver(bandit_10_arm)
    thompson_sampling(bandit_10_arm)




