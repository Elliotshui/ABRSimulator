import numpy as np

class SpeedController:
    # may need more info about max and min speed
    def __init__(self, simulator):
        self.simulator = simulator
        self.mpd = self.simulator.get_mpd()
        self.qoe_metric = self.simulator.get_qoe_metric()
        self.max_speed = 1.1
        self.min_speed = 0.9
        self.internal_rebuffer_weight = 5
        self.internal_latency_weight = 1
        self.latency_cap = 10
    # add functions you need here

    def get_next_speed(self, buffer_level, latency):
        # define sigmoid function: R -> (-1, 1)
        sigmoid = lambda x: 2 / (1 + np.exp(-x)) - 1
        # get weights
        w_buffer = self.qoe_metric.rebuffer_weight * self.internal_rebuffer_weight
        w_latency = self.qoe_metric.latency_weight * self.internal_latency_weight
        max_buffer = self.mpd.max_buffer
        # clip latency
        latency = self.latency_cap if latency > self.latency_cap else latency
        # calculate weighted sum
        sum = w_latency*latency - w_buffer*(max_buffer - buffer_level)
        # map sum to value between -1 and 1
        control = sigmoid(sum)
        #set speed midpoint
        bound = self.min_speed if control < 0 else self.max_speed
        speed = 1 + control * abs(bound - 1)
        print(speed)

        return speed
