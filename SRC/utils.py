import random
import matplotlib.pyplot as plt

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

