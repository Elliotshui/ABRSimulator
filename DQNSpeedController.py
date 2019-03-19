import numpy as np
import tensorflow as tf

class SpeedController:
    # may need more info about max and min speed
    def __init__(self, simulator):
        self.simulator = simulator
        self.mpd = self.simulator.get_mpd()
        self.qoe_metric = self.simulator.get_qoe_metric()
        self.max_speed = 1.1
        self.min_speed = 0.9

        self.sess = tf.Session()
        self.model = DQN(2, 3, 20)
        self.memory = Memory(100)
        self.memory_staging = {}

    # add functions you need here
    def get_next_speed(self, chunk_id, buffer_level, latency, previous_bitrates):
        memory_staging[chunk_id] = []
        state = (buffer_level, latency)

        speed = self.choose_speed(state)

        return speed

    def calc_reward(self, bitrate, prev_bitrate, rebuffer, latency

    def choose_speed(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.model.num_actions - 1)
        else:
            return np.argmax(self.model.predict_one(state, self.sess))

class DQN:
    def __init__(self, num_states, num_actions, batch_size):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        # define the placeholders
        self.actions = None
        # the output operations
        self.logits = None
        self.optimizer = None
        self.var_init = None
        # now setup the model
        self.setup_neural_net()

    def setup_neural_net(self):
        # setup placeholders
        self.states = tf.placeholder(shape=[None, self.num_states], dtype=tf.float32)
        self.action_values = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self.states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        self.logits = tf.layers.dense(fc2, self.num_actions)
        loss = tf.losses.mean_squared_error(self.action_values, self._logits)
        self.optimizer = tf.train.AdamOptimizer().minimize(loss)
        self.var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states:
                                                     state.reshape(1, self.num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self.states: x_batch, self.action_values: y_batch})

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self._max_memory:
            self.samples.pop(0)

    def sample(self, nosamples):
        if nosamples > len(self.samples):
            return random.sample(self.samples, len(self.samples))
        else:
            return random.sample(self.samples, nosamples)
