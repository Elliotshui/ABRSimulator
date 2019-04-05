import numpy as np
import random
from Simulator import Chunk
from Simulator import MPD

class MPDGenerator:
    def __init__(self, minimum_bitrate, maximum_bitrate):
        self.minimum_bitrate = minimum_bitrate
        self.maximum_bitrate = maximum_bitrate

    def generate_mpd(self, num_bitrates, video_length, chunk_length, max_buffer,
                     start_up_length):
        base_rates = np.linspace(
            self.minimum_bitrate, self.maximum_bitrate, num_bitrates+1)

        chunks = []
        for i in range(video_length):
            bitrates = np.zeros((num_bitrates))
            for i in range(num_bitrates):
                interval = base_rates[i+1] - base_rates[i]
                r = random.random() * interval
                bitrates[i] = base_rates[i] + r
            chunks.append(Chunk(bitrates))

        return MPD(video_length, chunk_length, max_buffer, start_up_length, chunks)
