import random
from collections import defaultdict

class FirstVisitMonteCarlo:
    def __init__(self, env, gamma=0.99, eps=0.1):
        self.env = env
        self.gamma = gamma
        self.eps = eps
        self.returns_sum = defaultdict(lambda: [0.0]*8)
        self.returns_count = defaultdict(lambda: [0]*8)
        self.Q = defaultdict(lambda: [0.0]*8)

    def policy(self, state):
        if random.random() < self.eps:
            return random.randrange(8)
        q = self.Q[state]
        return max(range(8), key=lambda a: q[a])

    def generate_episode(self, max_steps=1000):
        episode = []  # list of (state, action, reward)
        state = self.env.reset()
        for _ in range(max_steps):
            a = self.policy(state)
            next_state, r, done, _ = self.env.step(a)
            episode.append((state, a, r))
            state = next_state
            if done:
                break
        return episode

    def learn_episode(self):
        episode = self.generate_episode()
        G = 0.0
        visited = set()
        # iterate backwards
        for t in reversed(range(len(episode))):
            state, a, r = episode[t]
            G = self.gamma * G + r
            key = (state, a)
            if key not in visited:
                visited.add(key)
                self.returns_sum[state][a] += G
                self.returns_count[state][a] += 1
                self.Q[state][a] = self.returns_sum[state][a] / self.returns_count[state][a]
        total_reward = sum(r for (_,_,r) in episode)
        return total_reward

if __name__ == '__main__':
    from guardian_env import GuardianEnv
    env = GuardianEnv()
    agent = FirstVisitMonteCarlo(env)
    for ep in range(20):
        r = agent.learn_episode()
        print('Episode',ep,'return',r)