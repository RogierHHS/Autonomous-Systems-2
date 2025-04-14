#Vizdoom
from vizdoom import DoomGame, Mode  
import cv2
import numpy as np

class VizDoomEnvironment:
    # Functie die we callen wanneer we de environment starten
    def __init__(self, render=False, scenario="basic.cfg"):
        self.game = DoomGame()
        self.game.load_config(f"ViZDoom/scenarios/{scenario}")

        # Maken van render logica
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        # Starten van de game
        self.game.init()

        # Zelf gedefinieerde eigenschappen in plaats van gym.spaces
        self.observation_shape = (100, 160, 1)  # De shape van de image die we krijgen
        self.num_actions = 3                    # De acties die we kunnen doen in de game, in dit geval 3: [left, right, shoot]
        self.actions = [[1, 0, 0],  # Left
                        [0, 1, 0],  # Right
                        [0, 0, 1]]  # Shoot

    # Dit is hoe we een stap nemen in de environment
    def step(self, action):
        # Specificeer de actie en neem de stap
        reward = self.game.make_action(self.actions[action])  # De reward die we terug krijgen van de game

        # Overige informatie die we returnen
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            info = self.game.get_state().game_variables  # De game variables die we terug krijgen van de game
        else:
            state = np.zeros(self.observation_shape)
            info = 0

        done = self.game.is_episode_finished()

        return state, reward, done, info

    # Wat er gebeurt als we een nieuwe episode starten
    def reset(self):
        self.game.new_episode()  # Start een nieuwe episode
        state = self.game.get_state().screen_buffer  # Geeft de state terug van de game
        return self.grayscale(state)  # Geeft de grayscale image terug

    # Grayscale de game image/frame en resize de image naar 160x100
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)  # Zet de image om naar grayscale
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1)) 
        return state 

    # Call de close functie om de game af te sluiten
    def close(self):
        self.game.close()  # Sluit de game af
