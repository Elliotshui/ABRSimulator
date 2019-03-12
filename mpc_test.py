import mpc

def test_predict_throughput():
    throughput_values = [1, 2, 3, 4]
    throughput_times = [0, 1, 2, 3]
    horizon = 3
    C = mpc.predict_throughput(horizon, throughput_values,
                               method="harmonic")
    # TODO Improve test.
    return len(C) == horizon


class Chunk:
    def __init__(self, bitrates, sizes):
        self.bitrates = bitrates
        self.sizes = sizes

class MPD:
    def __init__(self, video_length, chunk_length, max_buffer, chunks):
        self.video_length = video_length
        self.chunk_length = chunk_length
        self.max_buffer = max_buffer
        self.chunks = chunks

class QOEMetric:
    def __init__(self, rebuffer_weight, variance_weight, startup_weight):
        self.rebuffer_weight = rebuffer_weight
        self.variance_weight = variance_weight
        self.startup_weight = startup_weight

class ChunkInfo:
    def __init__(self, chunk_number, previous_bitrate, previous_bandwidths,
                 buffer_level):
        self.chunk_number = chunk_number
        self.previous_bitrate = previous_bitrate
        self.previous_bandwidths = previous_bandwidths
        self.buffer_level = buffer_level

class VideoPlayer:
    def __init__(self, mpd, qoe, chunk_info):
        self.mpd = mpd
        self.qoe = qoe
        self.chunk_info = chunk_info
        self.abr = None
    def get_mpd(self):
        return self.mpd
    def get_qoe_metric(self):
        return self.qoe
    def get_next_chunk_info(self):
        return self.chunk_info

def init_player():
    chunks = []
    default_bitrates = [1, 2.5, 5, 8]
    default_sizes = default_bitrates.copy()

    numchunks = 60
    for n in range(numchunks):
        new_chunk = Chunk(default_bitrates.copy(), default_sizes.copy())
        chunks.append(new_chunk)

    mpd = MPD(numchunks, 1, 20, chunks)
    qoe = QOEMetric(1, 0, 0)
    previous_bandwidths = [2, 2.5, 4, 6, 8]
    chunk_info = ChunkInfo(20, 1, previous_bandwidths, 20)
    return VideoPlayer(mpd, qoe, chunk_info)

def init_test_mpc_abr():
    player = init_player()
    abr = mpc.MPCBitrateController(player)
    abr.horizon = 5
    return abr

def test_default_bitrate_utility():
    test_bitrate = 2.5
    abr = init_test_mpc_abr()
    utility = abr.bitrate_utility(test_bitrate)
    print("Test utility: {}".format(utility))
    # TODO Assert statement

def test_next_bitrate():
    abr = init_test_mpc_abr()
    next_bitrate = abr.next_bitrate()
    print("Test next bitrate: {}".format(next_bitrate))

test_next_bitrate()
