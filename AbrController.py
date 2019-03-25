import numpy as np
import tensorflow as tf
import tflearn
import Simulator
import a3c


class AbrController:
    def __init__(self):
        self.simulator = None
        self.mpd = None
        self.qoe_metric = None
        self.A_DIM = 5  # assume there is five bitrate options
        self.S_INFO = 6
        self.S_LEN = 8
        self.ACTOR_LR_RATE = 0.0001
        self.CRITIC_LR_RATE = 0.001
        self.TRAIN_SEQ_LEN = 100
        self.MODEL_SAVE_INTERVAL = 100
        self.CHUNK_TIL_VIDEO_END_CAP = 16.0
        self.GRADIENT_BATCH_SIZE = 16
        self.RAND_RANGE = 100000

        # normalization factors
        self.BITRATE_NORM = 1080
        self.BUFFER_NORM = 4

        action_vec = np.zeros(self.A_DIM)
        action_vec[0] = 1
        self.epoch = 0
        self.s_batch = [np.zeros((self.S_INFO, self.S_LEN))]
        self.r_batch = []
        self.a_batch = [action_vec]

        self.actor_gradient_batch = []
        self.critic_gradient_batch = []
        self.entropy_record = []

        self.sess = tf.Session()
        self.actor = a3c.ActorNetwork(self.sess, state_dim = [self.S_INFO, self.S_LEN], action_dim = self.A_DIM, learning_rate = self.ACTOR_LR_RATE)
        self.critic = a3c.CriticNetwork(self.sess, state_dim = [self.S_INFO, self.S_LEN], learning_rate = self.CRITIC_LR_RATE)


    def set_env(self, simulator):
        self.simulator = simulator
        self.qoe_metric = simulator.get_qoe_metric()
        self.mpd = simulator.get_mpd()

    def get_next_bitrate(self, chunk_id, previous_bitrates, previous_bandwidths, previous_download_times, buffer_level, rebuf):
        # calculate reward
        chunk_til_end = self.mpd.video_length - chunk_id
        video_bitrate = 0
        last_video_bitrate = 0
        if chunk_id <= 1:
            video_bitrate = self.mpd.chunks[0].bitrate[0]
            last_video_bitrate = self.mpd.chunks[0].bitrate[0]
        else:
            video_bitrate = self.mpd.chunks[chunk_id -
                                            1].bitrate[previous_bitrates[chunk_id - 1]]
            last_video_bitrate = self.mpd.chunks[chunk_id -
                                                 2].bitrate[previous_bitrates[chunk_id - 2]]
        previous_bandwidth = 0
        previous_download_time = 0
        if chunk_id == 0:
            previous_bandwidth = 0
            previous_download_time = 0
        else:
            previous_bandwidth = previous_bandwidths[chunk_id - 1]
            previous_download_time = previous_download_times[chunk_id - 1]
        reward = self.qoe_metric.bitrate_weight * video_bitrate + \
            self.qoe_metric.rebuffer_weight * rebuf + \
            self.qoe_metric.variance_weight * \
            np.abs(video_bitrate - last_video_bitrate)
        self.r_batch.append(reward)

        # set state
        state = np.array(self.s_batch[-1], copy=True)
        state = np.roll(state, -1, axis=1)

        state[0, -1] = video_bitrate / self.BITRATE_NORM
        state[1, -1] = buffer_level / self.BUFFER_NORM
        state[2, -1] = previous_bandwidth / 1000
        state[3, -1] = previous_download_time / self.BUFFER_NORM
        state[4, :self.A_DIM] = np.array(
            self.mpd.chunks[chunk_id].bitrates) / self.BITRATE_NORM
        state[5, -1] = np.minimum(chunk_til_end,
                                  self.CHUNK_TIL_VIDEO_END_CAP) / self.CHUNK_TIL_VIDEO_END_CAP

        action_prob = self.actor.predict(np.reshape(state, (1, self.S_INFO, self.S_LEN)))
        action_cumsum = np.cumsum(action_prob)
        bit_rate = (action_cumsum > np.random.randint(1, self.RAND_RANGE) / float(self.RAND_RANGE)).argmax()

        self.entropy_record.append(a3c.compute_entropy(action_prob[0]))

        if len(self.r_batch) >= self.TRAIN_SEQ_LEN:
            actor_gradient, critic_gradient, td_batch = \
                a3c.compute_gradients(s_batch=np.stack(self.s_batch[1:], axis=0),  # ignore the first chuck
                                      a_batch=np.vstack(self.a_batch[1:]),
                                      r_batch=np.vstack(self.r_batch[1:]),
                                      terminal=False, actor=self.actor, critic=self.critic)
            td_loss = np.mean(td_batch)

            self.actor_gradient_batch.append(actor_gradient)
            self.critic_gradient_batch.append(critic_gradient)

            self.entropy_record = []        
            if len(self.actor_gradient_batch) >= self.GRADIENT_BATCH_SIZE:

                assert len(self.actor_gradient_batch) == len(
                        self.critic_gradient_batch)
                    # assembled_actor_gradient = actor_gradient_batch[0]
                    # assembled_critic_gradient = critic_gradient_batch[0]
                    # assert len(actor_gradient_batch) == len(critic_gradient_batch)
                    # for i in xrange(len(actor_gradient_batch) - 1):
                    #     for j in xrange(len(actor_gradient)):
                    #         assembled_actor_gradient[j] += actor_gradient_batch[i][j]
                    #         assembled_critic_gradient[j] += critic_gradient_batch[i][j]
                    # actor.apply_gradients(assembled_actor_gradient)
                    # critic.apply_gradients(assembled_critic_gradient)

                for i in xrange(len(self.actor_gradient_batch)):
                    self.actor.apply_gradients(
                            self.actor_gradient_batch[i])
                    self.critic.apply_gradients(
                            self.critic_gradient_batch[i])

                self.actor_gradient_batch = []
                self.critic_gradient_batch = []

                self.epoch += 1
                if self.epoch % self.MODEL_SAVE_INTERVAL == 0:
                    # Save the neural net parameters to disk.
                    save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                               str(epoch) + ".ckpt")
                    print("Model saved in file: %s" % save_path)

            del self.s_batch[:]
            del self.a_batch[:]
            del self.r_batch[:]
