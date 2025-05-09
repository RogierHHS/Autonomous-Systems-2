import random
import matplotlib.pyplot as plt
from skimage import transform
from collections import deque
import numpy as np
from vizdoom import DoomGame

# Functie om één episode uit te voeren en frames op te slaan waar de actie is uitgevoerd
def collect_frames_with_actions(env, max_steps=60):
    """
    Verzamelt frames, acties en beloningen door een random agent maximaal `max_steps` stappen te laten uitvoeren.

    Parameters:
    env: De omgeving waarin acties worden uitgevoerd.
    max_steps (int): Maximum aantal stappen dat de agent mag zetten.

    Returns:
    list: Een lijst met tuples van (state, reward, step, action).
    """
    frames = []
    state = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = random.randint(0, env.num_actions - 1)
        state, reward, done, info = env.step(action)
        frames.append((state, reward, steps, action))
        steps += 1

    return frames


def show_shoot_frames(frames, target_action=2):
    """
    Laat alleen die frames zien waarin de opgegeven actie is uitgevoerd (bijv. schieten).

    Parameters:
    frames (list): Lijst met verzamelde (state, reward, step, action)-tuples.
    target_action (int): Index van de te tonen actie (bijv. 2 voor 'SHOOT').
    """
    actie_labels = ["LEFT", "RIGHT", "SHOOT"]
    
    for state, reward, step, action in frames:
        if action == target_action:
            plt.figure(figsize=(5, 3))
            plt.imshow(state.squeeze(), cmap='gray')
            plt.title(f"Stap {step} - Reward: {reward} - Actie: {actie_labels[action]}")
            plt.axis('off')
            plt.show()


def preprocess_frame(frame):
    """
    Verkleint het beeld naar 84x84 pixels en normaliseert het naar de range [0, 1].

    Parameters:
    frame (np.ndarray): Origineel beeld.

    Returns:
    np.ndarray: Voorverwerkt beeld met float32 waarden tussen 0 en 1.
    """
    preprocessed_frame = transform.resize(frame, (84, 84), mode='constant')
    return preprocessed_frame.astype(np.float32) / 255.0


def stack_frames(stacked_frames, state, is_new_episode):
    """
    Stapelt 4 frames op elkaar voor gebruik als input voor een neuraal netwerk.

    Parameters:
    stacked_frames (deque): Bestaande stack met eerdere frames.
    state (np.ndarray): Nieuwe observatie van de omgeving.
    is_new_episode (bool): Of het een nieuwe episode betreft.

    Returns:
    tuple: (stacked_state, updated stacked_frames)
    """
    # Verwijder singleton-dimensie als die aanwezig is
    state = np.squeeze(state, axis=-1) if state.ndim == 3 and state.shape[-1] == 1 else state

    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for i in range(4)], maxlen=4)
        for _ in range(4):
            stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


def create_environment(render=False):
    """
    Initialiseert de ViZDoom-omgeving met het 'basic' scenario en definieert alle mogelijke acties.

    Parameters:
    render (bool): Of het scherm zichtbaar moet zijn.

    Returns:
    tuple: (game object, lijst met actievectoren)
    """
    game = DoomGame()
    game.load_config(f"ViZDoom/scenarios/basic.cfg")
    game.set_doom_scenario_path(f"ViZDoom/scenarios/basic.wad")
    game.init()

    # Acties: bewegen, kijken, schieten, etc.
    links = [1, 0, 0, 0, 0, 0, 0]
    rechts = [0, 1, 0, 0, 0, 0, 0]
    links_kijk = [0, 0, 1, 0, 0, 0, 0]
    rechts_kijk = [0, 0, 0, 1, 0, 0, 0]
    schieten = [0, 0, 0, 0, 1, 0, 0]
    vooruit = [0, 0, 0, 0, 0, 1, 0]
    achteruit = [0, 0, 0, 0, 0, 0, 1]
    actions = [links, rechts, links_kijk, rechts_kijk, schieten, vooruit, achteruit]
    
    return game, actions


import gym
from gym import spaces
import numpy as np
