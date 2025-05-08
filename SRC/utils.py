import random
import matplotlib.pyplot as plt
import numpy as np

### Functie om één episode uit te voeren en frames op te slaan waar de actie is uitgevoerd ###
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

#### Functie om de frames te visualiseren waar de actie is uitgevoerd ###

def show_shoot_frames(frames, target_action=2):
    actie_labels = ["LEFT", "RIGHT", "SHOOT"]
    
    for state, reward, step, action in frames:
        if action == target_action:
            plt.figure(figsize=(5, 3))
            plt.imshow(state.squeeze(), cmap='gray')
            plt.title(f"Stap {step} - Reward: {reward} - Actie: {actie_labels[action]}")
            plt.axis('off')
            plt.show()

### visualisatie van Q-waarden ###

def visualize_q_table(agent, sample_state=None):
    print(f"Aantal geleerde states: {len(agent.q_table)}")

    # 1. Histogram van alle Q-waarden
    all_q_values = [q for q_vals in agent.q_table.values() for q in q_vals]
    plt.figure(figsize=(6, 4))
    plt.hist(all_q_values, bins=30, color='steelblue')
    plt.xlabel("Q-waarde")
    plt.ylabel("Frequentie")
    plt.title("Verdeling van Q-waarden na training")
    plt.show()

    # 2. Verdeling van gekozen beste acties
    action_counts = np.zeros(agent.num_actions)
    for q_vals in agent.q_table.values():
        best_action = np.argmax(q_vals)
        action_counts[best_action] += 1

    plt.figure(figsize=(6, 4))
    plt.bar(range(agent.num_actions), action_counts, color='darkorange')
    plt.xlabel("Actie index")
    plt.ylabel("Aantal keer gekozen als beste")
    plt.title("Verdeling van beste acties per state")
    plt.show()

    # 3. Q-waarden voor een sample state (als meegegeven)
    if sample_state is not None:
        sample_key = agent.get_state_key(sample_state)
        q_vals = agent.q_table[sample_key]
        plt.figure(figsize=(6, 4))
        plt.bar(range(agent.num_actions), q_vals, color='mediumseagreen')
        plt.xlabel("Actie")
        plt.ylabel("Q-waarde")
        plt.title("Q-waarden voor voorbeeldstate")
        plt.show()
    else:
        print("Geen voorbeeldstate meegegeven → sla individuele Q-plot over.\n")

    # 4. Top 5 meest ‘zekere’ beslissingen
    print("Top 5 meest overtuigende Q-beslissingen (grootste verschil tussen beste actie en gemiddeld):")
    confidences = []
    for key, q_vals in agent.q_table.items():
        diff = np.max(q_vals) - np.mean(q_vals)
        confidences.append((diff, key, q_vals))

    top_confident = sorted(confidences, reverse=True)[:5]
    for i, (score, key, q_vals) in enumerate(top_confident):
        print(f"Top {i+1}:")
        print(f"  Q-waarden: {np.round(q_vals, 2)}")
        print(f"  Beste actie: {np.argmax(q_vals)} met vertrouwen {score:.2f}\n")