from deep_q_environment import VizDoomEnvironment
import random

def random_baseline(env, episodes=5):
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
    env = VizDoomEnvironment(render=True)
    random_baseline(env, episodes=10)
    env.close()