import numpy as np

class BitrateController:
    """ Adaptive bitrate controller implementing BOLA algorithm

    -Quality of experience paremeters, required on init-
    rebuffer_weight : float indicating importance of minimizing rebuffering
    bitrate_utility : function mapping bitrate index to utility value

    -Required inputs for each video-
    allowed_bitrates : list of average chunk bitrates per bitrate category
    bitrate_sizes : list of average chunk sizes per bitrate category (or just
                    calculate by multiplying bitrate by chunk_length?)
    chunk_length : video chunk length in seconds
    video_length : total video length in seconds (or chunks?)
    max_buffer : maximum allowed buffer in seconds (or chunks?)

    -Required inputs for each chunk-
    buffer_level : current state of buffer in seconds (or chunks?)
    current_playback_time : in seconds (or chunks?)
    previous_bitrate_index : chosen bitrate index of previous chunks
    previous_bandwith : measured (average?) bandwith when downloading previous
                        chunk

    --Output per chunk--
    next_bitrate : which bitrate to download
    """
    def __init__(self, player=None, bitrate_utility=None):
        """Initialize ABR controller with player state."""
        self.name = "BOLA Bitrate"
        if player:
            self.player = player
            self.mpd = player.get_mpd()
            self.qoe = player.get_qoe_metric()
        self.bitrate_utility = self.default_bitrate_utility

    def assign_simulator(self, simulator):
        self.player = simulator
        self.mpd = self.player.get_mpd()
        self.qoe = self.player.get_qoe_metric()

    def set_player(self, player):
        """Set the associated video player object used for fetching data."""
        self.player = player

    def update_mpd(self):
        """Discard old MPD and get new from VideoPlayer"""
        self.mpd = self.player.get_mpd()

    def update_qoe(self):
        """Discard old QoE metric and get new from VideoPlayer"""
        self.qoe = self.player.get_qoe()

    def default_bitrate_utility(self, bitrate):
        """Identity function, return input bitrate 1:1"""
        return bitrate

    def log_bitrate_utility(self, chunk, bitrate):
        """Return bitrate utility based on a log scale"""
        ratio = bitrate / self.mpd.chunks[chunk].bitrates[-1]
        return np.log(ratio)

    def calc_buffer_control(self, chunk):
        """Return corresponding buffer control parameter value"""
        bitrates = self.mpd.chunks[chunk].bitrates
        playback_time = chunk*self.mpd.chunk_length
        time_to_end = self.mpd.video_length - playback_time

        t = min((playback_time, time_to_end))
        tprime = max((t/2, 3*self.mpd.chunk_length))
        max_buffer = min(self.mpd.max_buffer, tprime/self.mpd.chunk_length)

        utility1 = self.bitrate_utility(bitrates[0])
        gammap = self.qoe.rebuffer_weight * self.mpd.chunk_length
        return (max_buffer - 1)/(utility1 + gammap)

    def objective(self, chunk, bitrate_index, buffer_level, buffer_control):
        """Return value of QoE function"""
        bitrate = self.mpd.chunks[chunk].bitrates[bitrate_index]
        chunk_size = bitrate*self.mpd.chunk_length
        sum = 0
        sum += self.bitrate_utility(bitrate)
        sum += self.qoe.rebuffer_weight * self.mpd.chunk_length
        sum *= buffer_control
        sum -= buffer_level
        sum /= chunk_size
        return sum

    def chunk_number_to_time(self, chunk):
        """Return playback start time of chunk."""
        return chunk*self.mpd.chunk_length

    def suggest_bitrate(self, chunk, buffer_level):
        """Return the available bitrate that maximizes QoE function for the chunk."""
        #print("Chunk: " + str(chunk))
        bitrates = self.mpd.chunks[chunk].bitrates
        buffer_control = self.calc_buffer_control(chunk)

        M = len(bitrates)
        objective_sums = [
            self.objective(chunk, m, buffer_level, buffer_control) for m in range(M)]

        max_objective_index = objective_sums.index(max(objective_sums))
        return max_objective_index


    def refine_bitrate(self, chunk, new_bitrate_index, previous_bitrate_index,
                       previous_bandwith):
        """Return refined bitrate."""
        bitrates =  self.mpd.chunks[chunk].bitrates
        sizes = [bitrate*self.mpd.chunk_length
                 for bitrate
                 in self.mpd.chunks[chunk].bitrates]
        target_rate = max(previous_bandwith,
                                bitrates[-1]/self.mpd.chunk_length)


        minimum_found = False
        alt_bitrate_index = len(bitrates)-1
        while not minimum_found:
            size_alt = sizes[alt_bitrate_index]
            if size_alt/self.mpd.chunk_length <= target_rate:
                minimum_found = True
            else:
                alt_bitrate_index -= 1

        if alt_bitrate_index >= new_bitrate_index:
            alt_bitrate_index = new_bitrate_index
        elif alt_bitrate_index < previous_bitrate_index:
            alt_bitrate_index = previous_bitrate_index
        else:
            alt_bitrate_index = alt_bitrate_index + 1

        return alt_bitrate_index

    def get_next_bitrate(
            self, chunk_id, previous_bitrates,
            previous_bandwidths, buffer_level, *args):
        """Get next chunk info from player and return next bitrate to download."""
        if previous_bitrates:
            previous_bitrate_index = previous_bitrates[-1]
        else:
            previous_bitrate_index = 0
        if previous_bandwidths:
            previous_bandwidth = previous_bandwidths[-1]
        else:
            previous_bandwidth = 0

        new_bitrate_index = self.suggest_bitrate(chunk_id, buffer_level)
        if new_bitrate_index > previous_bitrate_index:
            new_bitrate_index = self.refine_bitrate(
                chunk_id, new_bitrate_index,
                previous_bitrate_index, previous_bandwidth)

        return new_bitrate_index
