import random
import math
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, alpha=0.5, gamma=0.99, eps=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q = defaultdict(lambda: [0.0]*8)

    def policy(self, state):
        # epsilon-greedy
        if random.random() < self.eps:
            return random.randrange(8)
        q = self.Q[state]
        return max(range(8), key=lambda a: q[a])

    def learn_episode(self, max_steps=1000):
        state = self.env.reset()
        total_reward = 0
        for _ in range(max_steps):
            a = self.policy(state)
            next_state, r, done, _ = self.env.step(a)
            total_reward += r
            best_next = max(self.Q[next_state])
            # Q-learning update
            self.Q[state][a] += self.alpha * (r + self.gamma * best_next - self.Q[state][a])
            state = next_state
            if done:
                break
        return total_reward

if __name__ == '__main__':
    from guardian_env import GuardianEnv
    env = GuardianEnv()
    agent = QLearningAgent(env)
    for ep in range(20):
        r = agent.learn_episode()
        print('Episode',ep,'return',r)