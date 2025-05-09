import random
import matplotlib.pyplot as plt
import numpy as np

### 1. Verzamelen van frames met bijbehorende acties ###
def collect_frames_with_actions(env, max_steps=60):
    frames = []
    state, _ = env.reset()  # ✅ aangepaste unpacking
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = random.randint(0, env.num_actions - 1)
        state, reward, done, info, _ = env.step(action)  # ✅ aangepaste unpacking
        frames.append((state, reward, steps, action))
        steps += 1

    return frames

### 2. Visualiseren van frames met specifieke actie ###
def show_shoot_frames(frames, target_action=2):
    actie_labels = ["LEFT", "RIGHT", "SHOOT"]
    
    for state, reward, step, action in frames:
        if action == target_action:
            plt.figure(figsize=(5, 3))
            # ✅ Check shape voor grayscale/kleur
            if state.ndim == 3 and state.shape[-1] == 1:
                plt.imshow(state.squeeze(), cmap='gray')
            elif state.ndim == 3 and state.shape[0] == 3:
                plt.imshow(np.moveaxis(state, 0, -1))  # channels-first → channels-last
            else:
                plt.imshow(state)  # fallback
            plt.title(f"Stap {step} - Reward: {reward} - Actie: {actie_labels[action]}")
            plt.axis('off')
            plt.show()

### 3. Q-table visualisatie ###
def visualize_q_table(agent, sample_state=None):
    print(f"Aantal geleerde states: {len(agent.q_table)}")

    # Histogram van alle Q-waarden
    all_q_values = [q for q_vals in agent.q_table.values() for q in q_vals]
    plt.figure(figsize=(6, 4))
    plt.hist(all_q_values, bins=30, color='steelblue')
    plt.xlabel("Q-waarde")
    plt.ylabel("Frequentie")
    plt.title("Verdeling van Q-waarden na training")
    plt.show()

    # Verdeling van gekozen beste acties
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

    # Q-waarden voor voorbeeldstate
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

    # Top 5 meest overtuigende beslissingen
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
