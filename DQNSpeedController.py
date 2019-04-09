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
STATE_LEN = 4  # Number of previous bandwidths to take into account
MAX_EPS = 0.99  # Initial epsilon for e-greedy policy
MIN_EPS = 0.001  # Minimum epsilon for e-greedy policy
LAMBDA = 0.0005  # Epsilon decay rate
GAMMA = 0.99  # Future reward discounting factor
BATCH_SIZE = 32  # Number of states to include in experience replay
MAX_MEMORY = 1000000  # Maximum number of experienced states to store in memory

# Overridable speed controller parameters
MAX_SPEED = 1.1  # Maximum allowed playback speed
MIN_SPEED = 0.9  # Minimum allowed playback speed
NUM_SPEEDS = 3  # Number of discrete speeds to use


class SpeedController(DQNController.Controller):
    """Video playback speed controller live-streaming implementing DQN.

    Attributes:
        simulator: Simulator object that will use the speed controller.
        mpd: MPD object describing the video to be played.
        qoe_metric: QOEMEtric object containing QoE parameters.
        max_speed: float representing maximum allowed playback speed
        min_speed: float representing minimum allowed playback speed
        num_speeds: int - number of discrete speeds
        speeds: array containing allowed speeds
        max_eps: initial epsilon for e-greedy policy
        min_eps: minimum episolon for e-greedy policy
        lmb: epsilon decay rate
        eps: current value of epsilon
        state_len: number of timesteps DQN should consider
        state_dim: number of dimension in state, should not be changed
        model: DQN object representing neural network
        memory: Memory object containing sampled states
        sess: tf.Session object


    """
    def __init__(self, simulator, max_speed=MAX_SPEED, min_speed=MIN_SPEED,
                 num_speeds=NUM_SPEEDS, max_eps=MAX_EPS, min_eps=MIN_EPS,
                 lmb=LAMBDA, max_memory=MAX_MEMORY, batch_size=BATCH_SIZE,
                 state_len=STATE_LEN, state_dim=STATE_DIM, num_runs=None,
                 gamma=GAMMA):
        self.name = "DQN Speed"
        self.training_mode = True
        self.dummy_output = True

        self.simulator = simulator
        self.mpd = self.simulator.get_mpd()
        self.qoe_metric = self.simulator.get_qoe_metric()

        self.num_runs = num_runs
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.num_speeds = num_speeds
        self.speeds = np.linspace(self.min_speed, self.max_speed, self.num_speeds)
        self.max_eps = max_eps
        self.min_eps = min_eps

        if lmb is None:
            self.lmb = self.calc_lmb(self.num_runs)
        else:
            self.lmb = lmb

        self.eps = self.max_eps
        self.gamma = gamma

        self.state_len = state_len
        self.state_dim = state_dim
        self.model = DQN(state_dim, num_speeds, batch_size, state_len)
        self.memory = DQNController.Memory(max_memory)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(self.model.var_init)
        self.memory_staging = {}  # Used to match actions with outcomes
        self.steps = 0

    def assign_simulator(self, simulator):
        self.simulator = simulator
        self.mpd = self.simulator.get_mpd()
        self.qoe_metric = self.simulator.get_qoe_metric()

    def reset_eps(self):
        self.eps = self.max_eps
        self.steps = 0

    def set_lmb(self, num_runs):
        self.lmb = self.calc_lmb(num_runs)

    def calc_lmb(self, num_runs):
        lmb = (- np.log((self.min_eps + 0.0005 - self.min_eps)
               / (self.max_eps - self.min_eps))
               / (num_runs*len(self.mpd.chunks)))
        return lmb

    def get_next_speed(self, chunk_id, buffer_level, latency,
                       rebuffer, previous_bitrate, buffered_bitrates,
                       previous_bandwidths):
        """Return next speed according to e-greedy policy and train network."""
        if self.dummy_output:
            return 1

        time_start = time.time()
        # Operates at playback time. Playback bitrates is list of previously selected
        # bitrates, including the bitrate of chunk_id.
        #state = np.array((buffer_level, latency)) Primitive state

        state = self.create_state_array(
            buffer_level, latency, previous_bitrate, buffered_bitrates, previous_bandwidths)

        if not self.training_mode:
            return self.speeds[np.argmax(self.model.predict_one(state, self.sess))]

        if chunk_id > 1:
            # Calculate reward for previous action and add this to the
            # previously saved list in the memory_staging dict.
            bitrate1 = previous_bitrate
            if len(buffered_bitrates) == 0:
                bitrate2 = bitrate1
            else:
                bitrate2 = buffered_bitrates[0]
            previous_reward = self.calc_reward(bitrate1, bitrate2, rebuffer, latency)
            self.memory_staging[chunk_id - 1].extend((previous_reward, state))
            self.memory.add_sample(tuple(self.memory_staging[chunk_id-1]))
            del self.memory_staging[chunk_id - 1]

            # Train DQN using saved experiences.
            self.replay()

        speed_index = self.choose_speed(state)

        # Save current state and action as extendable list
        self.memory_staging[chunk_id] = [state, speed_index]

        # Decay epsilon value for e-greedy policy.
        self.steps += 1
        self.eps = (self.min_eps + (self.max_eps - self.min_eps)
                                 * math.exp(-self.lmb * self.steps))

        elapsed_time = time_start - time.time()
        #print("Get_next_speed runtime: {}, time per minute: {}".format(elapsed_time, elapsed_time*60))
        return self.speeds[speed_index]

    def create_state_array(self, buffer_level, latency, previous_bitrate,
                           buffered_bitrates, previous_bandwidths):
        """Return array representing state.
        Array has the form:
            [buffer_level       0       0       ...     0           ]
            [latency_list       0       0       ...     0           ]
            [previous_bitrate   0       0       ...     0           ]
            [bb0                bb+1    bb+2    ...     bb+state_dim]
            [bw0                bw-1    bw-2    ...     bw-state_dim]
        Where bb = buffered_bitrates and bw = previous_bandwidths
        """
        state = np.zeros((self.state_dim, self.state_len))
        state[0, 0] = buffer_level
        state[1, 0] = latency
        state[2, 0] = previous_bitrate

        if len(buffered_bitrates) < self.state_len:
            state[3, :len(buffered_bitrates)] = buffered_bitrates[:]
        else:
            state[3, :] = buffered_bitrates[:self.state_len]

        if len(previous_bandwidths) < self.state_len:
            state[4, :len(previous_bandwidths)] = previous_bandwidths
        else:
            state[4, :] = previous_bandwidths[len(previous_bandwidths)-self.state_len:]
        return state

    def choose_speed(self, state):
        """Choose next speed for given state according to e-greedy policy."""
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
        self.states = tf.placeholder(
            shape=[None, self.num_states, self.state_len], dtype=tf.float32)
        self.action_values = tf.placeholder(
            shape=[None, self.num_actions], dtype=tf.float32)
        # Action values should be calculated as r + Q(s', a')

        # Split input into three parts
        buffer_and_latency = self.states[:, 0:2, 0]
        playback_bitrates = self.states[:, 2:3, :]
        previous_bandwidths = self.states[:, 3:4, :]

        # Treat the three parts individually
        fc1 = tf.layers.dense(buffer_and_latency, 50, activation=tf.nn.relu)
        cnn1 = tflearn.conv_1d(playback_bitrates, 128, 4, activation='relu')
        cnn2 = tflearn.conv_1d(previous_bandwidths, 128, 4, activation='relu')
        flat1 = tf.layers.flatten(cnn1)
        flat2 = tf.layers.flatten(cnn2)

        # Merge the three parts into a single input vector
        full_net = tf.concat([fc1, flat1, flat2], 1)
        fc2 = tf.layers.dense(full_net, 50, activation=tf.nn.relu)
        self.logits = tf.layers.dense(fc2, self.num_actions)

        # Define loss using difference between initial prediction and updated
        # prediction after an action has been taken and reward given.
        loss = tf.losses.mean_squared_error(self.action_values, self.logits)
        self.optimizer = tf.train.AdamOptimizer().minimize(loss)

        self.var_init = tf.global_variables_initializer()
