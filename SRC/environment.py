from vizdoom import DoomGame, Mode
import cv2
import numpy as np

import gym
from gym import spaces

class VizDoomEnvironment:
    def __init__(self, render=False, scenario="basic.cfg", actions=None):
        self.render_mode = render
        self.scenario = scenario
        self.game = DoomGame()
        self.game.load_config(f"ViZDoom/scenarios/{self.scenario}")
        self.game.set_window_visible(self.render_mode)
        self.game.init()

        # Allow custom actions or use default actions
        self.actions = actions if actions is not None else [
    [1, 0, 0, 0, 0, 0, 0],  # Naar links
    [0, 1, 0, 0, 0, 0, 0],  # Naar rechts
    [0, 0, 1, 0, 0, 0, 0],  # Kijk naar links
    [0, 0, 0, 1, 0, 0, 0],  # Kijk naar rechts
    [0, 0, 0, 0, 1, 0, 0],  # Alleen schieten
    [0, 0, 0, 0, 0, 1, 0],  # Vooruit
    [0, 0, 0, 0, 0, 0, 1],  # Achteruit
]
        self.num_actions = len(self.actions)  # Add num_actions attribute
        self.action_space = spaces.Discrete(self.num_actions)

        # Define observation space
        screen_height, screen_width = 84, 84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(screen_height, screen_width, 1), dtype=np.uint8
        )

    # Other methods remain unchanged
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        if state is not None:
            # Transpose the state to (height, width, channels)
            state = np.transpose(state, (1, 2, 0))
        return self._preprocess(state)

    def step(self, action):
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()
        state = self.game.get_state().screen_buffer if not done else None
        if state is not None:
            state = np.transpose(state, (1, 2, 0))
        return self._preprocess(state), reward, done, {}

    def render(self, mode="human"):
        pass  # Optional: Implement rendering logic if needed

    def close(self):
        self.game.close()

    def seed(self, seed=None):
        np.random.seed(seed)

    def _preprocess(self, state):
        if state is None:
            # Return a blank frame if the state is None
            return np.zeros((84, 84, 1), dtype=np.uint8)
        # Convert the state to grayscale and resize to 84x84
        gray_state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        resized_state = cv2.resize(gray_state, (84, 84))
        return np.expand_dims(resized_state, axis=-1)
