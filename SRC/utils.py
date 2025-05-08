import random
import matplotlib.pyplot as plt
from skimage import transform
from collections import deque
import numpy as np
from vizdoom import DoomGame

# Functie om één episode uit te voeren en frames op te slaan waar de actie is uitgevoerd
def collect_frames_with_actions(env, max_steps=60):
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
    actie_labels = ["LEFT", "RIGHT", "SHOOT"]
    
    for state, reward, step, action in frames:
        if action == target_action:
            plt.figure(figsize=(5, 3))
            plt.imshow(state.squeeze(), cmap='gray')
            plt.title(f"Stap {step} - Reward: {reward} - Actie: {actie_labels[action]}")
            plt.axis('off')
            plt.show()

def preprocess_frame(frame):
    # Resize the frame to 84x84 and convert to grayscale
    preprocessed_frame = transform.resize(frame, (84, 84), mode='constant')
    return preprocessed_frame.astype(np.float32) / 255.0  # Normalize to [0, 1] range

def stack_frames(stacked_frames, state, is_new_episode):

    # Remove singleton dimension if it exists
    state = np.squeeze(state, axis=-1) if state.ndim == 3 and state.shape[-1] == 1 else state


    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for i in range(4)], maxlen=4)
        for _ in range(4):  # Append the same frame 4 times
            stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames
def create_environment(render=False):
    game = DoomGame()
    game.load_config(f"ViZDoom/scenarios/basic.cfg")

    game.set_doom_scenario_path(f"ViZDoom/scenarios/basic.wad")

    game.init()

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

class VizDoomGymWrapper(gym.Env):
    def __init__(self, vizdoom_env):
        super(VizDoomGymWrapper, self).__init__()
        self.vizdoom_env = vizdoom_env

        # Use the action and observation spaces from the underlying environment
        self.action_space = self.vizdoom_env.action_space
        self.observation_space = self.vizdoom_env.observation_space

    def reset(self):
        # Reset the underlying VizDoom environment
        return self.vizdoom_env.reset()

    def step(self, action):
        # Step through the underlying VizDoom environment
        obs, reward, done, info = self.vizdoom_env.step(action)
        return obs, reward, done, info

    def render(self, mode="human"):
        # Render the underlying VizDoom environment
        self.vizdoom_env.render(mode)

    def close(self):
        # Close the underlying VizDoom environment
        self.vizdoom_env.close()

    def seed(self, seed=None):
        # Set the random seed for the environment
        if hasattr(self.vizdoom_env, "seed"):
            self.vizdoom_env.seed(seed)
        np.random.seed(seed)