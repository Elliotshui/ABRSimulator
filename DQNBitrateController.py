import numpy as np
import tensorflow as tf
import random
import tflearn
import math
import time

import DQNController

STATE_DIM = 5  # Number of state 'dimensions'. This is not a tunable param and
               # must be set to the actual number of 'state dimensions'

# Overridable DQN hyperparameters
STATE_LEN = 5  # Number of previous bandwidths to take into account
MAX_EPS = 0.99  # Initial epsilon for e-greedy policy
MIN_EPS = 0.001  # Minimum epsilon for e-greedy policy
LAMBDA = 0.5  # Epsilon decay rate
GAMMA = 0.99  # Future reward discounting factor
BATCH_SIZE = 32  # Number of states to include in experience replay
MAX_MEMORY = 1000000  # Maximum number of experienced states to store in memory

class BitrateController(DQNController.Controller):
    def __init__(self, simulator,
                 max_eps=MAX_EPS, min_eps=MIN_EPS, gamma=GAMMA,
                 lmb=LAMBDA, max_memory=MAX_MEMORY, batch_size=BATCH_SIZE,
                 state_len=STATE_LEN, state_dim=STATE_DIM, num_runs=None):
        self.simulator = simulator
        self.mpd = self.simulator.get_mpd()
        self.qoe_metric = self.simulator.get_qoe_metric()
        self.num_bitrates = len(self.mpd.chunks[0].bitrates)

        self.num_runs = num_runs
        self.max_eps = max_eps
        self.min_eps = min_eps
        if lmb is None:
            self.lmb = -np.log((min_eps + 0.0005 - self.min_eps)/(self.max_eps - self.min_eps))/(self.num_runs*len(self.mpd.chunks))
        else:
            self.lmb = lmb

        self.eps = self.max_eps
        self.gamma = gamma

        self.state_len = state_len
        self.state_dim = state_dim
        self.model = DQN(state_dim, self.num_bitrates, batch_size, state_len)
        self.memory = DQNController.Memory(max_memory)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(self.model.var_init)
        self.sess.run(self.model.local_init)
        self.memory_staging = {}  # Used to match actions with outcomes
        self.steps = 0

        self.time_tracker = []


    # add functions you need here
    def get_next_bitrate(self, chunk_id, previous_bitrates, previous_bandwidths,
                         buffer_level, latency, rebuffer):
        """Return next speed according to e-greedy policy and train network."""
        self.time_tracker = []
        self.time_tracker.append(time.time())
        # Operates at playback time. Playback bitrates is list of previously selected
        # bitrates, including the bitrate of chunk_id.
        #state = np.array((buffer_level, latency)) Primitive state
        if len(previous_bitrates) != 0:
            previous_bitrate = previous_bitrates[-1]
        else:
            previous_bitrate = 0

        allowed_bitrates = np.array(self.mpd.chunks[chunk_id].bitrates)
        state = self.create_state_array(
            buffer_level, latency, previous_bitrate,
            previous_bandwidths, allowed_bitrates)

        if chunk_id > 1:
            # Calculate reward for previous action and add this to the
            # previously saved list in the memory_staging dict.
            bitrate_index1 = previous_bitrates[-1]
            bitrate_index2 = previous_bitrates[-2]
            bitrate1 = self.mpd.chunks[chunk_id-1].bitrates[bitrate_index1]
            bitrate2 = self.mpd.chunks[chunk_id-2].bitrates[bitrate_index2]
            previous_reward = self.calc_reward(bitrate1, bitrate2, rebuffer, latency)
            self.memory_staging[chunk_id - 1].extend((previous_reward, state))
            self.memory.add_sample(tuple(self.memory_staging[chunk_id-1]))
            del self.memory_staging[chunk_id - 1]
            # Train DQN using saved experiences.
            self.replay()
        #self.time_tracker.append(time.time())
        #print("Before choose_bitrate: {}".format(self.time_tracker[-1] - self.time_tracker[-2]))
        bitrate_index = self.choose_bitrate(state)
        # Save current state and action as extendable list
        self.memory_staging[chunk_id] = [state, bitrate_index]

        # Decay epsilon value for e-greedy policy.
        self.steps += 1
        self.eps = (self.min_eps + (self.max_eps - self.min_eps)
                                 * math.exp(-self.lmb * self.steps))

        self.time_tracker.append(time.time())
        elapsed_time = self.time_tracker[-1] - self.time_tracker[-2]
        #print("Get_next_bitrate runtime: {}, time per minute: {}".format(elapsed_time, elapsed_time*60))
        return bitrate_index

    def create_state_array(self, buffer_level, latency, previous_bitrate, previous_bandwidths, bitrates):
        """Return array representing state.
        Array has the form:
            [buffer_level       0       0       ...     0           ]
            [latency            0       0       ...     0           ]
            [previous_bitrate   0       0       ...     0           ]
            [bw0                bw-1    bw-2    ...     bw-state_dim]
        Where and bw = previous_bandwidths
        """
        if self.num_bitrates <= self.state_len:
            state = np.zeros((self.state_dim, self.state_len))
        else:
            state = np.zeros((self.state_dim, self.num_bitrates))

        state[0, 0] = buffer_level
        state[1, 0] = latency
        state[2, 0] = previous_bitrate

        if len(previous_bandwidths) < self.state_len:
            state[3, :len(previous_bandwidths)] = previous_bandwidths
        else:
            state[3, :] = previous_bandwidths[len(previous_bandwidths)-self.state_len:]

        state[4, :self.num_bitrates] = bitrates

        return state

    def choose_bitrate(self, state):
        """Choose next bitrate for given state according to e-greedy policy."""
        if random.random() < self.eps:
            return random.randint(0, self.model.num_actions - 1)
        else:
            return np.argmax(self.model.predict_one(state, self.sess))

    def save_model(self, id):
        # Save the variables to disk.
        save_path = self.saver.save(self.sess, "tmp/"+id+".ckpt")
        print("Model saved in path: %s" % save_path)

class DQN(DQNController.DQN):
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
        # Setup model
        self.setup_neural_net()

    def setup_neural_net(self):
        """Sets up the neural network and loss function."""
        if self.state_len < self.num_states:
            state_height = self.num_actions
        else:
            state_height = self.state_len
        self.states = tf.placeholder(
            shape=[None, self.num_states, state_height], dtype=tf.float32)
        self.action_values = tf.placeholder(
            shape=[None, self.num_actions], dtype=tf.float32)
        # Action values should be calculated as r + Q(s', a')

        print(self.num_actions)
        # Split input into three parts
        scalars = self.states[:, 0:3, 0]
        previous_bandwidths = self.states[:, 3:4, :]
        available_bitrates = self.states[:, 4:5, :self.num_actions]

        # Treat the three parts individually
        fc1 = tf.layers.dense(scalars, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(available_bitrates, 50, activation=tf.nn.relu)
        cnn1 = tflearn.conv_1d(previous_bandwidths, 128, 4, activation='relu')
        flat1 = tf.layers.flatten(fc2)
        flat2 = tf.layers.flatten(cnn1)

        # Merge the three parts into a single input vector
        full_net = tf.concat([fc1, flat1, flat2], 1)
        fc3 = tf.layers.dense(full_net, 50, activation=tf.nn.relu)
        self.logits = tf.layers.dense(fc3, self.num_actions)

        # Define loss using difference between initial prediction and updated
        # prediction after an action has been taken and reward given.
        loss = tf.losses.mean_squared_error(self.action_values, self.logits)
        self.optimizer = tf.train.AdamOptimizer().minimize(loss)
        self.var_init = tf.global_variables_initializer()
        self.local_init = tf.local_variables_initializer()
