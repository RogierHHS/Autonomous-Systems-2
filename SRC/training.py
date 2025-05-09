from agent import QLearningAgent

def train_q_learning(env, episodes=100,
                     learning_rate=0.1,
                     gamma=0.99,
                     epsilon=1.0,
                     epsilon_decay=0.995,
                     epsilon_min=0.01,
                     frame_skip=1,
                     policy="greedy"):  
    """
    Traint een Q-learning agent in de opgegeven omgeving.

    Parameters:
    env: De omgeving waarin de agent leert.
    episodes (int): Aantal episodes om te trainen.
    learning_rate (float): Hoe snel de Q-waarden worden aangepast.
    gamma (float): Discount factor voor toekomstige beloningen.
    epsilon (float): Startwaarde voor epsilon-greedy exploratie.
    epsilon_decay (float): Hoe snel epsilon afneemt na elke episode.
    epsilon_min (float): Minimale waarde van epsilon.
    frame_skip (int): Aantal stappen dat dezelfde actie wordt herhaald.
    policy (str): Actiekeuzebeleid, bijv. 'greedy'.

    Returns:
    tuple: (getrainde agent, lijst met rewards per episode)
    """
    
    agent = QLearningAgent(
        num_actions=env.num_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        policy=policy 
    )

    reward_history = []

    for ep in range(episodes):
        total_reward = 0

        # Ondersteunt zowel reset() → state als reset() → (state, enemy_position)
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state, enemy_position = reset_result
        else:
            state, enemy_position = reset_result, None

        state_key = agent.get_state_key(state, enemy_position)
        done = False

        while not done:
            action = agent.choose_action(state_key)

            cumulative_reward = 0
            for _ in range(frame_skip):
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, done, _, enemy_position = step_result
                else:
                    next_state, reward, done, _ = step_result
                    enemy_position = None

                cumulative_reward += reward
                if done:
                    break

            next_state_key = agent.get_state_key(next_state, enemy_position)
            agent.learn(state_key, action, cumulative_reward, next_state_key, done)

            state_key = next_state_key
            total_reward += cumulative_reward

        reward_history.append(total_reward)
        print(f"Episode {ep+1} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    return agent, reward_history

