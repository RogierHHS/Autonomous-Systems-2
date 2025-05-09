import numpy as np
import cv2
import random
from collections import defaultdict

class QLearningAgent:
    """
    Een Q-learning agent die discrete acties leert op basis van 
    observaties uit de omgeving. Ondersteunt zowel greedy als softmax beleid.

    Parameters:
    num_actions (int): Aantal mogelijke acties.
    learning_rate (float): Leerpercentage voor Q-updates.
    gamma (float): Discount factor voor toekomstige beloningen.
    epsilon (float): Startwaarde voor exploratie bij epsilon-greedy.
    epsilon_decay (float): Afnamefactor van epsilon per episode.
    epsilon_min (float): Minimale waarde van epsilon.
    policy (str): Actiekeuzebeleid: 'greedy' of 'softmax'.
    """

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

    def get_state_key(self, state, enemy_position=None):
        """
        Verkleint en versimpelt de staat tot een hashbare key voor gebruik in de Q-table.

        Parameters:
        state (np.ndarray): De inputobservatie.
        enemy_position (str | None): Optionele extra info over vijandpositie.

        Returns:
        tuple: Hashbare sleutel voor gebruik in de Q-table.
        """
        reduced = cv2.resize(state.squeeze(), (20, 10), interpolation=cv2.INTER_AREA)
        binned = (reduced // 64).astype(int)
        state_flat = tuple(binned.flatten())
        return (state_flat, enemy_position) if enemy_position is not None else state_flat

    def choose_action(self, state_key):
        """
        Kiest een actie op basis van het huidige beleid.

        Parameters:
        state_key (tuple): De key van de huidige staat.

        Returns:
        int: De gekozen actie-index.
        """
        q_values = self.q_table[state_key]

        if self.policy == "greedy":
            if random.random() < self.epsilon:
                return random.randint(0, self.num_actions - 1)
            return int(np.argmax(q_values))

        elif self.policy == "softmax":
            tau = max(self.epsilon, 0.1)
            q_shifted = q_values - np.max(q_values)  # voorkomt overflow
            exp_q = np.exp(q_shifted / tau)
            probs = exp_q / np.sum(exp_q)
            return np.random.choice(len(q_values), p=probs)

        else:
            raise ValueError(f"Onbekende policy '{self.policy}'. Kies 'greedy' of 'softmax'.")

    def learn(self, state_key, action, reward, next_state_key, done):
        """
        Voert een Q-update uit op basis van ontvangen beloning.

        Parameters:
        state_key (tuple): Huidige staat.
        action (int): Uitgevoerde actie.
        reward (float): Ontvangen beloning.
        next_state_key (tuple): Volgende staat.
        done (bool): Of de episode is beëindigd.
        """
        q_current = self.q_table[state_key][action]
        q_next = 0 if done else np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.lr * (reward + self.gamma * q_next - q_current)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

