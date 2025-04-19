from vizdoom import DoomGame, Mode
import cv2
import numpy as np

class VizDoomEnvironment:
    # Functie die we callen wanneer we de environment starten
    def __init__(self, render=False, scenario="basic.cfg", actions=None):
        self.game = DoomGame()
        self.game.load_config(f"ViZDoom/scenarios/{scenario}")

        self.game.set_window_visible(render)
        self.game.init()

        # Acties instellen
        if actions is None:
            # Standaard acties (voor basic.cfg): LEFT, RIGHT, SHOOT
            self.actions = [
                [1, 0, 0],  # LEFT
                [0, 1, 0],  # RIGHT
                [0, 0, 1]   # SHOOT
            ]
        else:
            self.actions = actions

        self.num_actions = len(self.actions)
        self.observation_shape = (100, 160, 1)  # De shape van de image die we krijgen

    # Dit is hoe we een stap nemen in de environment
    def step(self, action):
        reward = self.game.make_action(self.actions[action])  # De reward die we terug krijgen van de game

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            info = self.game.get_state().game_variables
        else:
            state = np.zeros(self.observation_shape)
            info = 0

        done = self.game.is_episode_finished()
        return state, reward, done, info

    # Wat er gebeurt als we een nieuwe episode starten
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)

    # Grayscale de game image/frame en resize de image naar 160x100
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resize, (100, 160, 1))

    # Call de close functie om de game af te sluiten
    def close(self):
        self.game.close()
