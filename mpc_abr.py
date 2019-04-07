
import numpy as np
import itertools
#from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from scipy.optimize import brute

""" ALGORITHM PSEUDO CODE
initialize
for k in K_list:
    if player_startup:
        C_pred[k:k+N] = ThroughputPred(C[:k])
        R[k], T_s = f_st(R_[k-1], B[k], C_pred[k:k+N]) # solve QOE_MAX
        Start playback after T_s seconds
    else if playback_started:
        C_pred[k:k+N] = ThroughputPred(C[:k])
        R_k = f(R_[k-1], B[k], C_pred[k:k+N]) # solve QOE_MAX_STEADY
    download chunk k with bitrate R_k, wait until finished
"""

class BitrateController:
    """ Adaptive bitrate controller implementing BOLA algorithm

    -Required on init-
    variance_weight : float indicating importance of minimizing bitrate variance
    startup_weight : float indicating importance of minimizing startup time
    rebuffer_weight : float indicating importance of minimizing rebuffering
    bitrate_utility : function mapping bitrate index to utility value
    horizon : N, how many forward steps to optimize bitrate over

    -Required inputs for each video-
    allowed_bitrates : list of average chunk bitrates per bitrate category
    bitrate_sizes : list of average chunk sizes per bitrate category
    chunk_length : video chunk length in seconds
    video_length : total video length in seconds (or chunks?)
    max_buffer : maximum allowed buffer in seconds (or chunks?)

    -Required inputs for each chunk-
    buffer_level : current state of buffer in seconds (or chunks?)
    current_playback_time : in seconds (or chunks?)
    previous_bitrate_index : chosen bitrate index of previous chunks
    previous_bandwiths : measured (average?) bandwiths when downloading previous
                        N chunks
            OR
    forecasted_bandwidths : predicted bandwidths for the next N chunks (per
                            chunk or as function of time?)


    --Output per chunk--
    next_bitrate : which bitrate to download
    (pausing instructions?)
    """
    def __init__(self, player, bitrate_utility=None, horizon=None):
        """Initialize ABR controller with player state."""
        self.name = "MPC Bitrate"
        if player:
            self.player = player
            self.mpd = player.get_mpd()
            self.qoe = player.get_qoe_metric()
        self.bitrate_utility = self.default_bitrate_utility
        self.horizon = 3 if horizon is None else horizon

    def assign_simulator(self, simulator):
        self.player = simulator
        self.mpd = self.player.get_mpd()
        self.qoe = self.player.get_qoe_metric()

    def update_mpd():
        """Discard old MPD and get new from VideoPlayer"""
        self.mpd = self.player.get_mpd()

    def update_qoe():
        """Discard old QoE metric and get new from VideoPlayer"""
        self.qoe = self.player.get_qoe()

    def predict_throughput(self, horizon, throughput_values,
                           throughput_times=None, method="harmonic"):
        """Take bandwidth history and return predicted future bandwidths."""
        if method == "expsmoothing":
            data = np.array(throughput_values)
            model = SimpleExpSmoothing(data)
            model_fit = model.fit(0.5)
            pred_start = len(throughput_values)
            pred_end = pred_start + horizon - 1
            prediction = model_fit.predict(pred_start, pred_end)
            return prediction

        elif method == "harmonic":
            prediction = []
            for i in range(horizon):
                history_size = len(throughput_values)
                if history_size is 0:
                    return [100 for _ in range(horizon)]

                sum_inverse = 0
                for throughput in throughput_values:
                    sum_inverse += 1/throughput
                tp_prediction = history_size / sum_inverse
                prediction.append(tp_prediction)
                throughput_values.append(tp_prediction)
            return prediction

    def default_bitrate_utility(self, bitrate):
        """Identity function, return input bitrate 1:1"""
        return bitrate

    def log_bitrate_utility(self, chunk, bitrate):
        """Return bitrate utility based on a log scale"""
        ratio = bitrate / self.mpd.chunks[chunk].bitrates[-1]
        return np.log(ratio)

    def get_chunk_sizes(self, chunk):
            sizes = []
            for bitrate in self.mpd.chunks[chunk].bitrates:
                sizes.append(bitrate*self.mpd.chunk_length)
            return sizes

    def calc_wait(self, chunk, buffer_level, bitrate_index, bandwidth):
        """Return required wait time to avoid overfilling buffer."""
        chunk_size = self.get_chunk_sizes(chunk)[bitrate_index]
        new_buffer = max(0, buffer_level - chunk_size/bandwidth)
        wait_time = new_buffer + self.mpd.chunk_length - self.mpd.max_buffer
        return max(0, wait_time)

    def next_buffer(self, chunk, buffer_level, bitrate_index, bandwidth):
        """Return expected buffer level after chunk download."""
        chunk_size = self.get_chunk_sizes(chunk)[bitrate_index]
        wait_time = self.calc_wait(chunk, buffer_level,
                                   bitrate_index, bandwidth)
        temp_buffer = max(0, buffer_level - chunk_size/bandwidth)
        new_buffer = max(0, temp_buffer + self.mpd.chunk_length - wait_time)
        return new_buffer

    def objective(self, R_arg, chunk, previous_bitrate, predicted_bandwidths,
                  buffer_level):
        """Return value of QoE function over given future chunks."""
        R_arg = [int(r) for r in R_arg]
        horizon = len(R_arg)
        sizes = [
            self.get_chunk_sizes(i) for i in range(chunk, chunk+horizon)]
        bitrates = [
            self.mpd.chunks[i].bitrates for i in range(chunk, chunk+horizon)]

        bandwidths = predicted_bandwidths

        R = [previous_bitrate]
        R += R_arg

        buffer_vector = np.zeros(horizon)
        buffer_vector[0] = buffer_level

        video_quality = 0
        quality_variance = 0
        rebuffer_time = 0
        startup_delay = 0 #TODO Implement startup delay calc

        # QoE of video segments from k to k+N-1
        for i in range(0, horizon):
            # Average video quality
            video_quality += self.bitrate_utility(bitrates[i][R[i+1]])
            # Average quality variations
            quality_variance += abs(self.bitrate_utility(bitrates[i][R[i+1]])
                        - self.bitrate_utility(bitrates[i][R[i]]))
            # Rebuffer
            rebuffer_time += (max(0, sizes[i][R[i+1]], self.mpd.chunk_length)
                / bandwidths[i] - buffer_vector[i])
            dt = 0
            if i != horizon-1:
                buffer_vector[i+1] = self.next_buffer(
                    chunk, buffer_vector[i], R[i+1], bandwidths[i])

        qoe_sum = (video_quality - self.qoe.variance_weight*quality_variance
                                 - self.qoe.rebuffer_weight*rebuffer_time
                                 - self.qoe.startup_weight*startup_delay)
        #print(str(R) + ": " + str(qoe_sum))
        #print("Vidqual {},     rebuffer {}".format(video_quality, self.qoe.rebuffer_weight*rebuffer_time) )
        return -qoe_sum

    def optimize_qoe(self, chunk, previous_bitrate, predicted_bandwidths,
                     buffer_level, horizon):
        """Return optimal bitrates from current chunk through horizon."""
        num_rates = len(self.mpd.chunks[0].bitrates)
        choices = (slice(0, num_rates, 1),) * horizon
        bitrate_indices = range(0, num_rates)
        #choices = itertools.product(bitrate_indices, repeat=self.horizon)
        arg = (chunk, previous_bitrate, predicted_bandwidths, buffer_level)
        result = brute(self.objective, choices, args=arg, disp=True, finish=None)
        if not hasattr(result, '__iter__'):
            result = [result]
        return result

    def get_next_bitrate(
            self, chunk_id, previous_bitrates,
            previous_bandwidths, buffer_level, *args):
        """Take state data and return optimal bitrate for next chunk."""
        if (self.mpd.video_length/self.mpd.chunk_length - chunk_id) < self.horizon:
            horizon = int(self.mpd.video_length/self.mpd.chunk_length - chunk_id)
        else:
            horizon = self.horizon
        predicted_bandwidths = self.predict_throughput(self.horizon, previous_bandwidths)
        if previous_bitrates:
            previous_bitrate = previous_bitrates[-1]
        else:
            previous_bitrate = 0
        result = self.optimize_qoe(chunk_id, previous_bitrate,
                                   predicted_bandwidths, buffer_level, horizon)

        #print(self.mpd.video_length - chunk)

        #print(result[0])
        return int(result[0])
