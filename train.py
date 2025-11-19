import argparse
import numpy as np
import matplotlib.pyplot as plt
from guardian_env import GuardianEnv

parsers = argparse.ArgumentParser()
parsers.add_argument('--algo', choices=['qlearning','sarsa','monte'], default='qlearning')
parsers.add_argument('--episodes', type=int, default=500)
args = parsers.parse_args()

if args.algo == 'qlearning':
    from agents.q_learning import QLearningAgent as Agent
elif args.algo == 'sarsa':
    from agents.sarsa import SarsaAgent as Agent
else:
    from agents.monte_carlo import FirstVisitMonteCarlo as Agent

env = GuardianEnv()
agent = Agent(env)
returns = []
for ep in range(args.episodes):
    r = agent.learn_episode()
    returns.append(r)
    if (ep+1) % max(1,args.episodes//10) == 0:
        print(f'Episode {ep+1}/{args.episodes} return {r:.1f}')

# simple plot
plt.plot(np.convolve(returns, np.ones(25)/25, mode='valid'))
plt.title(f'{args.algo} learning curve')
plt.xlabel('Episode')
plt.ylabel('Smoothed return')
plt.show()