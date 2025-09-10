import numpy as np

Q = np.zeros((5, 2))   # states=5, actions=2
alpha, gamma, episodes = 0.1, 0.9, 1000

for _ in range(episodes):
    state = np.random.randint(0, 5)
    action = np.random.randint(0, 2)
    reward = np.random.randn()
    next_state = np.random.randint(0, 5)
    Q[state, action] = Q[state, action] + alpha * (
        reward + gamma * np.max(Q[next_state]) - Q[state, action])
