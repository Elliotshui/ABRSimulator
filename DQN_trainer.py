import Simulator
import AbrController
import SpeedController
import HeuristicSpeedController
import DQNSpeedController
import DQNBitrateController
import mpc_abr
import bola_abr
import trace_fetcher
import matplotlib.pyplot as plt
import sys
import time
import random
import numpy as np
import pickle
import mpd_generator
import os

# constants
INTERVAL = 0.5  # interval of network trace
CHUNK_LEN = 1   # default length for a chunk
MAX_BUFFER = 5
START_UP = 2

MPD_MINIMUM_BITRATE = 240
MPD_MAXIMUM_BITRATE = 2000


BITR_W = 1
REBUF_W = 1000
VAR_W = 1
START_W = 1
LAT_W = 1000

# Training settings
NUM_RUNS = 5
REAL_TRACES = False
SAVE_MODEL = False

# Override globals with command line arguments
if len(sys.argv) > 1:
    NUM_RUNS = int(sys.argv[1])
if len(sys.argv) > 2:
    BITR_W = float(sys.argv[2])
    REBUF_W = float(sys.argv[3])
    VAR_W = float(sys.argv[4])
    START_W = float(sys.argv[5])
    LAT_W = float(sys.argv[6])

class Trainer():
    def __init__(self):
        pass

    def main(self):
        # simulator initiation
        simulator = Simulator.Simulator()

        mpdfile = "trace/example_video"
        simulator.set_mpd(CHUNK_LEN, MAX_BUFFER, START_UP, mpdfile)

        simulator.display_plots = False

        #networktrace = "trace/example_networktrace"
        qoe_metric = Simulator.QOEMetric(BITR_W, REBUF_W, VAR_W, START_W, LAT_W)
        simulator.set_qoe_metric(qoe_metric)
        #simulator.set_network_info(INTERVAL, networktrace)

        # trace fetcher
        if REAL_TRACES:
            sim_trace_fetcher = trace_fetcher.TraceFetcher(
                real_traces = True,
                trace_folder_path = "C:\\Users\\Einar Lenneloev\\Desktop\\Traces\\cooked2",
                dt = 0.5
            )
        else:
            sim_trace_fetcher = trace_fetcher.TraceFetcher(
                dt = 0.5,  # Timestep in seconds
                maximum_bw = 2500,  # Maximum allowed bandwidth
                minimum_bw = 200,  # Minimum allowed bandwidth
                fine_variability = 0.1,  # Control fine variation amplitude
                course_variability = 0.6,  # Control course variation amplitude
                course_freq = 15,  # Control course variation frequency
                smoothing_factor = 0.1, # Control strength of smoothing
                post_noise = 10  # Standard deviation of post-noise
            )
        simulator.trace_fetcher = sim_trace_fetcher
        initial_trace = simulator.trace_fetcher.get_random_trace(200)
        simulator.network_info = Simulator.NetworkInfo(INTERVAL, initial_trace)

        # controller initiation
        #abr_controller = DQNBitrateController.BitrateController(
        #    simulator, lmb=None, num_runs=NUM_RUNS)
        #speed_controller = DQNSpeedController.SpeedController(
        #    simulator, lmb=None, num_runs=NUM_RUNS)
        abr_controller = DQNBitrateController.BitrateController(
            simulator)
        speed_controller = DQNSpeedController.SpeedController(
            simulator)

        # set controller
        simulator.abr_controller = abr_controller
        simulator.speed_controller = speed_controller

        qoe_vals = []
        run_list = []

        sub_save_count = 0

        for k in range(1, NUM_RUNS+1):
            #number_of_bitrates = random.randint(4, 8)
            mpd = self.generate_mpd(5, 60)
            simulator.mpd = mpd
            simulator.network_info.bandwidths = simulator.trace_fetcher.get_random_trace(200)
            time_tracker = []
            if k == NUM_RUNS:
                simulator.display_plots = False
            qoe = simulator.run()
            #print("Simulator run time: {}".format(time_tracker[-1] - time_tracker[-2]))
            if k % 1 == 0:
                if SAVE_MODEL:
                    simulator.abr_controller.save_model(self.id+"abr_"+str(sub_save_count))
                    simulator.speed_controller.save_model(self.id+"speed_"+str(sub_save_count))
                    sub_save_count += 1
                print(k/NUM_RUNS)
            qoe_val = qoe[0] - qoe[1] - qoe[2] - qoe[3] - qoe[4]
            qoe_vals.append(qoe_val)
            run_list.append(k)
            #print(qoe)
            #print(simulator.abr_controller.eps)
        #g_tracegen.plot_trace(timesteps, synthetic_bandwidths)

        #g_tracegen.plot_trace(timesteps, synthetic_bandwidths)

        #plt.plot(simulator.chunk_history[:-1], simulator.speed_history)
        #plt.show()

        #plt.plot(simulator.chunk_history, simulator.bitrate_history)
        #plt.plot(simulator.chunk_history, simulator.bandwidth_history)
        #plt.show()
        plt.plot(run_list, qoe_vals, '.')
        plt.title('Learning progress')
        plt.xlabel('Simulation run')
        plt.ylabel('QoE sum')
        plt.show()
        self.qoe_vals = qoe_vals
        self.run_list = run_list

        return simulator.abr_controller, simulator.speed_controller

    def generate_mpd_and_trace(self, num_bitrates, video_length):
        mpd_gen = mpd_generator.MPDGenerator(MPD_MINIMUM_BITRATE, MPD_MAXIMUM_BITRATE)
        mpd = mpd_gen.generate_mpd(
            num_bitrates, video_length, CHUNK_LEN, MAX_BUFFER, START_UP)
        trace = self.generate_trace(video_length)
        return mpd, trace

    def generate_mpd(self, num_bitrates, video_length):
        mpd_gen = mpd_generator.MPDGenerator(MPD_MINIMUM_BITRATE, MPD_MAXIMUM_BITRATE)
        mpd = mpd_gen.generate_mpd(
            num_bitrates, video_length, CHUNK_LEN, MAX_BUFFER, START_UP)
        return mpd

    def generate_trace(self, video_length):
        g_dt = 0.5  # Timestep in seconds
        g_length = video_length*100  # Length of trace in seconds
        g_maximum_bw = 2500  # Maximum allowed bandwidth
        g_minimum_bw = 200  # Minimum allowed bandwidth
        g_fine_variability = 0.1  # Control fine variation amplitude
        g_course_variability = 0.6  # Control course variation amplitude
        g_course_freq = 15  # Control course variation frequency
        g_smoothing_factor = 0.1 # Control strength of smoothing
        g_post_noise = 10  # Standard deviation of post-noise
        g_tracegen = trace_generator.TraceGenerator(0)

        timesteps, trace = g_tracegen.generate_trace(
            g_dt, g_length, g_maximum_bw, g_minimum_bw, g_fine_variability,
            g_course_variability, g_course_freq, g_post_noise, g_smoothing_factor)

        return trace


if __name__ == '__main__':
    trainer = Trainer()
    trainer.id = str(time.time())[-5:].replace('.', '')
    abr_controller, speed_controller = trainer.main()
    if SAVE_MODEL:
        pickle.dump(trainer, open("tmp/"+trainer.id+'.p', 'wb'))
        loaded_trainer = pickle.load(open("tmp/"+trainer.id+'.p', 'rb'))
        abr_controller.save_model(trainer.id+"abr_final")
        speed_controller.save_model(trainer.id+"speed_final")
