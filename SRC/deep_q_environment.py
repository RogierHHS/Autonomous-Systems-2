from vizdoom import DoomGame, Mode
import cv2
import numpy as np

import gym
from gym import spaces

class DeepVizDoomEnvironment:
    def __init__(self, render=False, scenario="basic.cfg", actions=None, use_grayscale=True):
        self.render_mode = render
        self.scenario = scenario
        self.use_grayscale = use_grayscale  
        self.game = DoomGame()
        self.game.load_config(f"ViZDoom/scenarios/{self.scenario}")
        self.game.set_window_visible(self.render_mode)
        self.game.init()

        self.actions = actions if actions is not None else [
            [1, 0, 0, 0, 0, 0, 0],  # Naar links
            [0, 1, 0, 0, 0, 0, 0],  # Naar rechts
            [0, 0, 1, 0, 0, 0, 0],  # Kijk naar links
            [0, 0, 0, 1, 0, 0, 0],  # Kijk naar rechts
            [0, 0, 0, 0, 1, 0, 0],  # Alleen schieten
            [0, 0, 0, 0, 0, 1, 0],  # Vooruit
            [0, 0, 0, 0, 0, 0, 1],  # Achteruit
        ]
        self.num_actions = len(self.actions) 
        self.action_space = spaces.Discrete(self.num_actions)

        # defineer de space
        screen_height, screen_width = 84, 84
        channels = 1 if self.use_grayscale else 3
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(screen_height, screen_width, channels), dtype=np.uint8
        )

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        if state is not None:
            state = np.transpose(state, (1, 2, 0))
        return self._preprocess(state)

    def step(self, action):
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()
        state = self.game.get_state().screen_buffer if not done else None
        if state is not None:
            state = np.transpose(state, (1, 2, 0))
        return self._preprocess(state), reward, done, {}

    def close(self):
        self.game.close()

    def seed(self, seed=None):
        np.random.seed(seed)

    def _preprocess(self, state):
        if state is None:
            # Return a blank frame if the state is None
            shape = (84, 84, 1) if self.use_grayscale else (84, 84, 3)
            return np.zeros(shape, dtype=np.uint8)
        if self.use_grayscale:
            # Convert the state to grayscale and resize to 84x84
            gray_state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            resized_state = cv2.resize(gray_state, (84, 84))
            return np.expand_dims(resized_state, axis=-1)
        else:
            # Resize the state to 84x84 without converting to grayscale
            resized_state = cv2.resize(state, (84, 84))
            return resized_state
