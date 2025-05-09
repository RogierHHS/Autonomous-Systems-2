from vizdoom import DoomGame
import numpy as np
import cv2

class VizDoomEnvironment:
    """
    Een ViZDoom-omgeving met optionele grijswaarden en vijand-detectie op basis van kleur.

    Parameters:
    render (bool): Of de omgeving visueel weergegeven moet worden.
    scenario (str): Naam van het te laden scenario-configuratiebestand.
    actions (list): Actiematrix (bijv. links, rechts, schieten). Als None wordt een standaardset gebruikt.
    use_grayscale (bool): Of de observatie wordt omgezet naar grijswaarden.
    use_enemy_color_detection (bool): Of vijandposities gedetecteerd worden op basis van kleurherkenning.
    """

    def __init__(self, render=False, scenario="basic.cfg", actions=None, use_grayscale=True, use_enemy_color_detection=False):
        self.game = DoomGame()
        self.game.load_config(f"ViZDoom/scenarios/{scenario}")
        self.game.set_window_visible(render)
        self.game.init()

        self.use_grayscale = use_grayscale
        self.use_enemy_color_detection = use_enemy_color_detection

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

    def detect_enemy_color(self, frame):
        """
        Detecteert vijandpositie op basis van vooraf gedefinieerde vijandskleuren.

        Parameters:
        frame (np.ndarray): Ruwe frame van het spel (channels-first formaat).

        Returns:
        str | None: Richting van vijand ('left', 'right', 'center') of None als geen vijand gevonden is.
        """
        enemy_colors_bgr = [
            (179, 0, 0), (33, 0, 1), (63, 63, 167), (0, 1, 71), (0, 255, 255),
            (44, 142, 186), (7, 15, 23), (29, 29, 114), (107, 155, 231), (143, 167, 191)
        ]
        return self.detect_enemy_position(frame, enemy_colors_bgr)

    def detect_enemy_position(self, frame, color_list, tolerance=30):
        """
        Bepaalt de richting van de vijand op basis van kleurdetectie.

        Parameters:
        frame (np.ndarray): Beeld met shape (C, H, W).
        color_list (list): Lijst met BGR-kleuren die als vijand gelden.
        tolerance (int): Acceptabele afwijking per kanaal bij kleurherkenning.

        Returns:
        str | None: Richting van de vijand ('left', 'right', 'center') of None.
        """
        img = np.moveaxis(frame, 0, -1)
        mask_total = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for bgr in color_list:
            lower = np.array([max(0, c - tolerance) for c in bgr], dtype=np.uint8)
            upper = np.array([min(255, c + tolerance) for c in bgr], dtype=np.uint8)
            mask = cv2.inRange(img, lower, upper)
            mask_total = cv2.bitwise_or(mask_total, mask)

        if cv2.countNonZero(mask_total) > 150:
            contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, _, w, _ = cv2.boundingRect(largest)
                center_x = x + w // 2
                width = img.shape[1]

                if center_x < width / 3:
                    return "left"
                elif center_x > 2 * width / 3:
                    return "right"
                else:
                    return "center"
        return None

    def step(self, action):
        """
        Voert een actie uit in de omgeving en retourneert observatie, reward en optioneel vijandpositie.

        Parameters:
        action (int): Index van de actie die uitgevoerd wordt.

        Returns:
        tuple: (state, reward, done, info, enemy_position)
        """
        reward = self.game.make_action(self.actions[action])

        if self.game.get_state():
            raw_frame = self.game.get_state().screen_buffer
            enemy_position = self.detect_enemy_color(raw_frame) if self.use_enemy_color_detection else None
            state = self.process_observation(raw_frame)
            info = self.game.get_state().game_variables
        else:
            state = np.zeros(self.observation_shape)
            enemy_position = None
            info = 0

        done = self.game.is_episode_finished()
        return state, reward, done, info, enemy_position

    def reset(self):
        """
        Zet de omgeving terug naar de start van een nieuwe episode.

        Returns:
        tuple: (eerste observatie, eventuele vijandpositie)
        """
        self.game.new_episode()
        raw_frame = self.game.get_state().screen_buffer
        enemy_position = self.detect_enemy_color(raw_frame) if self.use_enemy_color_detection else None
        state = self.process_observation(raw_frame)
        return state, enemy_position

    def process_observation(self, observation):
        """
        Verwerkt het ruwe beeld naar een geschikt formaat (optioneel grijs en resized).

        Parameters:
        observation (np.ndarray): Origineel beeld uit de omgeving.

        Returns:
        np.ndarray: Voorverwerkte observatie.
        """
        if self.use_grayscale:
            gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
            resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
            return np.reshape(resize, (100, 160, 1))
        else:
            img = np.moveaxis(observation, 0, -1)
            resized = cv2.resize(img, (160, 100), interpolation=cv2.INTER_CUBIC)
            return np.moveaxis(resized, -1, 0)

    def close(self):
        """
        Sluit de omgeving netjes af.
        """
        self.game.close()

