from environment import VizDoomEnvironment
import random

def random_baseline(env, episodes=5):
    """
    Voert een random baseline uit in de opgegeven omgeving.

    Parameters:
    env: De ViZDoom-omgeving waarin getest wordt.
    episodes (int): Het aantal episodes dat wordt uitgevoerd.

    Returns:
    list: Een lijst met totale rewards per episode.
    """
    print("Random baseline gestart...")
    total_scores = []

    for ep in range(episodes):
        score = 0
        state, _ = env.reset()  # Reset de omgeving; negeer tweede returnwaarde
        done = False

        while not done:
            action = random.randint(0, env.num_actions - 1)
            state, reward, done, info, _ = env.step(action)  # Negeer enemy_visible
            score += reward

        total_scores.append(score)
        print(f"Episode {ep+1}: totale reward = {score}")

    avg_score = sum(total_scores) / len(total_scores)
    print(f"\nGemiddelde reward over {episodes} episodes: {avg_score:.2f}")
    return total_scores

if __name__ == "__main__":
    env = VizDoomEnvironment(render=True)
    random_baseline(env, episodes=10)
    env.close()

