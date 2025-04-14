from agent import QLearningAgent

def train_q_learning(env, episodes=100,
                     learning_rate=0.1,
                     gamma=0.99,
                     epsilon=1.0,
                     epsilon_decay=0.995,
                     epsilon_min=0.01,
                     frame_skip=1):
    
    agent = QLearningAgent(
        num_actions=env.num_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min
    )

    reward_history = []

    for ep in range(episodes):
        total_reward = 0
        state = env.reset()
        state_key = agent.get_state_key(state)
        done = False

        while not done:
            action = agent.choose_action(state_key)

            # Actie uitvoeren met frame skipping
            cumulative_reward = 0
            for _ in range(frame_skip):
                next_state, reward, done, _ = env.step(action)
                cumulative_reward += reward
                if done:
                    break

            next_state_key = agent.get_state_key(next_state)
            agent.learn(state_key, action, cumulative_reward, next_state_key, done)

            state_key = next_state_key
            total_reward += cumulative_reward

        reward_history.append(total_reward)
        print(f"Episode {ep+1} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    return reward_history
