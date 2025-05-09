from vizdoom import DoomGame, Mode
import cv2
import numpy as np

import gym
from gym import spaces

class DeepVizDoomEnvironment:
    """
    Een aangepaste ViZDoom-omgeving voor gebruik met reinforcement learning.

    Parameters:
    render (bool): Geeft aan of het spel visueel weergegeven moet worden.
    scenario (str): Bestandsnaam van het scenario-configuratiebestand.
    actions (list): Lijst van actievectoren. Als None wordt een standaardset gebruikt.
    use_grayscale (bool): Bepaalt of observaties worden omgezet naar grijswaarden.
    """

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

        # Definieer de observatieruimte (shape van de staat)
        screen_height, screen_width = 84, 84
        channels = 1 if self.use_grayscale else 3
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(screen_height, screen_width, channels), dtype=np.uint8
        )

    def reset(self):
        """
        Zet de omgeving terug naar de start van een nieuwe episode.

        Returns:
        np.ndarray: De eerste observatie van de nieuwe episode.
        """
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        if state is not None:
            state = np.transpose(state, (1, 2, 0))  # Van (C, H, W) naar (H, W, C)
        return self._preprocess(state)

    def step(self, action):
        """
        Voert een actie uit in de omgeving.

        Parameters:
        action (int): Index van de te nemen actie.

        Returns:
        tuple: (nieuwe staat, reward, done, info)
        """
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()
        state = self.game.get_state().screen_buffer if not done else None
        if state is not None:
            state = np.transpose(state, (1, 2, 0))
        return self._preprocess(state), reward, done, {}

    def close(self):
        """
        Sluit het spel netjes af.
        """
        self.game.close()

    def seed(self, seed=None):
        """
        Zet de random seed voor reproduceerbaarheid.

        Parameters:
        seed (int): Seed-waarde voor numpy RNG.
        """
        np.random.seed(seed)

    def _preprocess(self, state):
        """
        Verwerkt de ruwe staat naar een genormaliseerde en consistente vorm.

        Parameters:
        state (np.ndarray | None): Ruwe staat van de omgeving.

        Returns:
        np.ndarray: Voorverwerkte staat (84x84, grijs of kleur).
        """
        if state is None:
            # Geen staat beschikbaar → geef lege afbeelding terug
            shape = (84, 84, 1) if self.use_grayscale else (84, 84, 3)
            return np.zeros(shape, dtype=np.uint8)
        if self.use_grayscale:
            # Converteer naar grijs en resize
            gray_state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            resized_state = cv2.resize(gray_state, (84, 84))
            return np.expand_dims(resized_state, axis=-1)
        else:
            # Resize kleurbeeld
            resized_state = cv2.resize(state, (84, 84))
            return resized_state

