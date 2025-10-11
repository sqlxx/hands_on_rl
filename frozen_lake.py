import gymnasium as gym
import time
import numpy as np

from dynamic_planning import PolicyIteration, ValueIteration, print_agent

def frozen_lake_demo(env, agent):
    observation, info = env.reset()
    print(f"初始状态: {observation}")
    env = env.unwrapped

    for action in env.P[14]:
        # print(action)
        print(env.P[14][action])

    try:
        while True:
            env.render()
            rnd = np.random.rand()
            policy = agent.pi[observation]
            for a in range(4):
                rnd -= policy[a]
                if rnd <= 0:
                    action = a
                    break
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExit")
    finally:
        env.close()

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
    unwrapped_env = env.unwrapped

    action_meaning = ['<', 'v', '>', '^']
    theta = 1e-5
    gamma = 0.9
    agent = PolicyIteration(unwrapped_env, theta, gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, [5, 7, 11,12], [15])

    # agent2 = ValueIteration(env, theta, gamma)
    # agent2.value_iteration()
    # print_agent(agent2, action_meaning, [5, 7, 11,12], [15])

    frozen_lake_demo(env, agent)