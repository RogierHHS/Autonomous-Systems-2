from deep_q_environment import DeepVizDoomEnvironment
import random

def deep_qrandom_baseline(env, episodes=5):
    """
    Voert een random baseline uit in de DeepVizDoom-omgeving.

    Parameters:
    env: De omgeving waarin het random gedrag getest wordt.
    episodes (int): Aantal episodes dat uitgevoerd wordt.

    Returns:
    tuple: Gemiddelde reward en een lijst met totale rewards per episode.
    """
    print("Random baseline gestart...")
    total_scores = []

    for ep in range(episodes):
        score = 0
        state = env.reset()  
        done = False

        while not done:
            action = random.randint(0, env.num_actions - 1)
            state, reward, done, info = env.step(action)
            score += reward

        total_scores.append(score)
        print(f"Episode {ep+1}: totale reward = {score}")

    avg_score = sum(total_scores) / len(total_scores)
    print(f"\nGemiddelde reward over {episodes} episodes: {avg_score:.2f}")
    return avg_score, total_scores

if __name__ == "__main__":
    env = DeepVizDoomEnvironment(render=True)
    random_baseline(env, episodes=10)
    env.close()
