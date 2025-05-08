from vizdoom import DoomGame, Mode
import cv2
import numpy as np

class VizDoomEnvironment:
    def __init__(self, render=False, scenario="basic.cfg", actions=None, use_grayscale=True):
        self.game = DoomGame()
        self.game.load_config(f"ViZDoom/scenarios/{scenario}")

        self.game.set_window_visible(render)
        self.game.init()

        self.use_grayscale = use_grayscale

        if actions is None:
            self.actions = [
                [1, 0, 0],  # LEFT
                [0, 1, 0],  # RIGHT
                [0, 0, 1]   # SHOOT
            ]
        else:
            self.actions = actions

        self.num_actions = len(self.actions)
        self.observation_shape = (100, 160, 1) if use_grayscale else (3, 240, 320)

    def step(self, action):
        reward = self.game.make_action(self.actions[action])

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.process_observation(state)
            info = self.game.get_state().game_variables
        else:
            state = np.zeros(self.observation_shape)
            info = 0

        done = self.game.is_episode_finished()
        return state, reward, done, info

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.process_observation(state)

    def process_observation(self, observation):
        if self.use_grayscale:
            gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
            resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
            return np.reshape(resize, (100, 160, 1))
        else:
            return observation

    def close(self):
        self.game.close()