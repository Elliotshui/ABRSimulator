
class SpeedController:
    # may need more info about max and min speed
    def __init__(self, simulator):
        self.simulator = simulator
        self.mpd = self.simulator.get_mpd()
        self.qoe_metric = self.simulator.get_qoe_metric()

    # add functions you need here   

    def get_next_speed(self):
        # modify your algorithm here
        return 1