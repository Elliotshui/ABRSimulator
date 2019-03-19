
class AbrController:
    def __init__(self, simulator):
        self.simulator = simulator
        self.mpd = self.simulator.get_mpd()
        self.qoe_metric = self.simulator.get_qoe_metric()

    # add functions you need here   

    def get_next_bitrate(self, chunk_id, previous_bitrates, previous_bandwidths, buffer_level):
        # modify your algorithm here
        return 0