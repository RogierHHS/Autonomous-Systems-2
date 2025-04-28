import numpy as np
class training():
    @staticmethod
    def predict_action(explore_start, explore_stop, decay_rate, episode, action_size):

        exp_exp_tradeoff = np.random.rand()
        exploration_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * episode)
        if (exploration_probability > exp_exp_tradeoff):
            action = np.random.choice(actions)
        
        else:
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs: state.reshape((1, *state.shape))})

            choice = np.argmax(Qs)
            action = actions[int(choice)]
    
        return action, exploration_probability

