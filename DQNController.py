import numpy as np
import tensorflow as tf
import random
import tflearn
import math
import time

class Controller:
    def __init__(self):
        pass

    def calc_reward(self, bitrate, prev_bitrate, partial_rebuffer, latency):
        """Calculate for reward according to QoE parameters."""
        reward = (self.qoe_metric.bitrate_weight*bitrate
                  - self.qoe_metric.rebuffer_weight*partial_rebuffer
                  - self.qoe_metric.variance_weight*abs(bitrate - prev_bitrate)
                  - self.qoe_metric.latency_weight*latency)
        return reward

    def replay(self):
        """Train network using experienced states stored in memory."""
        batch = self.memory.sample(self.model.batch_size)
        states = np.array([val[0] for val in batch])
        # TODO Change state retrieval
        next_states = np.array([(np.zeros(self.model.num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self.model.predict_batch(states, self.sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self.model.predict_batch(next_states, self.sess)
        # setup training arrays
        x = np.zeros((len(batch), self.model.num_states, self.state_len))
        y = np.zeros((len(batch), self.model.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + self.gamma * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        x = np.array(x)
        y = np.array(y)

        self.model.train_batch(self.sess, x, y)
        #print("Replay time: {}".format(time.time()-replay_start))

    

class DQN:
    """Represents the Deep Q Network."""
    def __init__(self, num_states, num_actions, batch_size, state_len):
        # Info about states and actions
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.state_len = state_len
        # Placeholders
        self.states = None
        self.actions = None
        # Output operations
        self.logits = None
        self.optimizer = None
        self.var_init = None

    def predict_one(self, state, sess):
        """Predict best next speed for a single given state."""
        return sess.run(self.logits, feed_dict={self.states:
                                                     state.reshape(1, self.num_states, self.state_len)})

    def predict_batch(self, states, sess):
        """Predict corresponding best speeds for a batch of states."""
        return sess.run(self.logits, feed_dict={self.states: states})

    def train_batch(self, sess, x_batch, y_batch):
        """Train network using batch of training data."""
        sess.run(self.optimizer, feed_dict={self.states: x_batch, self.action_values: y_batch})


class Memory:
    """Stores experienced states to be used in batch training."""
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.max_memory:
            self.samples.pop(0)

    def sample(self, nosamples):
        if nosamples > len(self.samples):
            return random.sample(self.samples, len(self.samples))
        else:
            return random.sample(self.samples, nosamples)
