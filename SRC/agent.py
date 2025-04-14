import numpy as np
import cv2
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.num_actions = num_actions
        self.lr = learning_rate                # α: hoeveel leer je per stap
        self.gamma = gamma                    # γ: hoe belangrijk is toekomstige reward
        self.epsilon = epsilon                # Startwaarde voor exploratie
        self.epsilon_decay = epsilon_decay    # Hoe snel exploratie afneemt
        self.epsilon_min = epsilon_min        # Minimale exploratie
        self.q_table = defaultdict(lambda: np.zeros(num_actions))  # Tabel met Q-waarden

    def get_state_key(self, state):
        # Downsample het beeld grofweg naar een lagere resolutie
        reduced = cv2.resize(state.squeeze(), (20, 10), interpolation=cv2.INTER_AREA)
        # Verdeel intensiteit in bins (grijswaarden van 0-255 → 0-3)
        binned = (reduced // 64).astype(int)
        return tuple(binned.flatten())


    def choose_action(self, state_key):
        # ε-greedy strategie: soms random actie kiezen, soms beste
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return int(np.argmax(self.q_table[state_key]))

    def learn(self, state_key, action, reward, next_state_key, done):
        q_current = self.q_table[state_key][action]
        q_next = 0 if done else np.max(self.q_table[next_state_key])

        # Q-learning update
        self.q_table[state_key][action] += self.lr * (reward + self.gamma * q_next - q_current)

        # Verlaag epsilon na elke episode
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
