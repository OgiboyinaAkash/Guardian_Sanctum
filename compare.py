import numpy as np
import matplotlib.pyplot as plt
from guardian_env import GuardianEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.monte_carlo import FirstVisitMonteCarlo

def run_algo(algo_cls, episodes=200):
    env = GuardianEnv()
    agent = algo_cls(env)
    returns = []
    for ep in range(episodes):
        r = agent.learn_episode()
        returns.append(r)
    return returns

if __name__ == '__main__':
    episodes = 300
    print('Running Q-Learning...')
    q = run_algo(QLearningAgent, episodes)
    print('Running SARSA...')
    s = run_algo(SarsaAgent, episodes)
    print('Running Monte Carlo...')
    m = run_algo(FirstVisitMonteCarlo, episodes)

    window = 25
    q_s = np.convolve(q, np.ones(window)/window, mode='valid')
    s_s = np.convolve(s, np.ones(window)/window, mode='valid')
    m_s = np.convolve(m, np.ones(window)/window, mode='valid')

    plt.plot(q_s, label='Q-Learning')
    plt.plot(s_s, label='SARSA')
    plt.plot(m_s, label='Monte Carlo')
    plt.legend()
    plt.title('Learning curves (smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed return')
    plt.show()