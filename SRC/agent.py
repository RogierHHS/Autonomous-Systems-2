import numpy as np
import cv2
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, policy="greedy"):
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.policy = policy.lower()
        self.q_table = defaultdict(lambda: np.zeros(num_actions))

    def get_state_key(self, state):
        reduced = cv2.resize(state.squeeze(), (20, 10), interpolation=cv2.INTER_AREA)
        binned = (reduced // 64).astype(int)
        return tuple(binned.flatten())

    def choose_action(self, state_key):
        q_values = self.q_table[state_key]

        if self.policy == "greedy":
            # Îµ-greedy policy
            if random.random() < self.epsilon:
                return random.randint(0, self.num_actions - 1)
            return int(np.argmax(q_values))

        elif self.policy == "softmax":
            # Softmax policy (Boltzmann exploration)
            tau = max(self.epsilon, 0.1)  # temperatuur
            q_shifted = q_values - np.max(q_values)  # voor stabiliteit
            exp_q = np.exp(q_shifted / tau)
            probs = exp_q / np.sum(exp_q)
            return np.random.choice(len(q_values), p=probs)

        else:
            raise ValueError(f"Onbekende policy '{self.policy}'. Kies 'greedy' of 'softmax'.")

    def learn(self, state_key, action, reward, next_state_key, done):
        q_current = self.q_table[state_key][action]
        q_next = 0 if done else np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.lr * (reward + self.gamma * q_next - q_current)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay