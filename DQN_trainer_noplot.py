import Simulator
import AbrController
import SpeedController
import HeuristicSpeedController
import DQNSpeedController
import DQNBitrateController
import trace_generator
import sys
import time



# constants
INTERVAL = 0.5  # interval of network trace
CHUNK_LEN = 1   # default length for a chunk
MAX_BUFFER = 5
START_UP = 2

BITR_W = 1
REBUF_W = 1000
VAR_W = 0
START_W = 0
LAT_W = 0

# Training settings
NUM_RUNS = 300

# Override globals with command line arguments
if len(sys.argv) > 1:
    NUM_RUNS = int(sys.argv[1])
if len(sys.argv) > 2:
    BITR_W = float(sys.argv[2])
    REBUF_W = float(sys.argv[3])
    VAR_W = float(sys.argv[4])
    START_W = float(sys.argv[5])
    LAT_W = float(sys.argv[6])

## DO NOT MODIFY THIS PART
# simulator initiation
simulator = Simulator.Simulator()

mpdfile = "trace/example_video"
simulator.set_mpd(CHUNK_LEN, MAX_BUFFER, START_UP, mpdfile)

# Generate network trace

g_dt = 0.5  # Timestep in seconds
g_length = 200  # Length of trace in seconds
g_maximum_bw = 3000  # Maximum allowed bandwidth
g_minimum_bw = 200  # Minimum allowed bandwidth
g_fine_variability = 0  # Control fine variation amplitude
g_course_variability = 0.3  # Control course variation amplitude
g_course_freq = 20  # Control course variation frequency
g_smoothing_factor = 0 # Control strength of smoothing
g_post_noise = 0  # Standard deviation of post-noise
g_tracegen = trace_generator.TraceGenerator(0)
timesteps, synthetic_bandwidths = g_tracegen.generate_trace(
    g_dt, g_length, g_maximum_bw, g_minimum_bw, g_fine_variability,
    g_course_variability, g_course_freq, g_post_noise, g_smoothing_factor)
simulator.network_info = Simulator.NetworkInfo(INTERVAL, synthetic_bandwidths)
#g_tracegen.plot_trace(timesteps, synthetic_bandwidths)

simulator.display_plots = False

#networktrace = "trace/example_networktrace"
qoe_metric = Simulator.QOEMetric(BITR_W, REBUF_W, VAR_W, START_W, LAT_W)
simulator.set_qoe_metric(qoe_metric)
#simulator.set_network_info(INTERVAL, networktrace)

# controller initiation
abr_controller = DQNBitrateController.BitrateController(
    simulator, lmb=None, num_runs=NUM_RUNS)
speed_controller = DQNSpeedController.SpeedController(
    simulator, lmb=None, num_runs=NUM_RUNS)

# set controller
simulator.abr_controller = abr_controller
simulator.speed_controller = speed_controller

qoe_vals = []
run_list = []

for k in range(1, NUM_RUNS+1):
    time_tracker = []
    if k == NUM_RUNS:
        simulator.display_plots = True
    print(k/NUM_RUNS)
    time_tracker.append(time.time())
    qoe = simulator.run()
    time_tracker.append(time.time())
    print("Simulator run time: {}".format(time_tracker[-1] - time_tracker[-2]))
    qoe_val = qoe[0] - qoe[1] - qoe[2] - qoe[3] - qoe[4]
    qoe_vals.append(qoe_val)
    run_list.append(k)
    #print(qoe)
    #print(simulator.abr_controller.eps)


#g_tracegen.plot_trace(timesteps, synthetic_bandwidths)

#plt.plot(simulator.chunk_history[:-1], simulator.speed_history)
#plt.show()

#plt.plot(simulator.chunk_history, simulator.bitrate_history)
#plt.plot(simulator.chunk_history, simulator.bandwidth_history)
#plt.show()
#plt.plot(run_list, qoe_vals, '.')
#plt.title('Learning progress')
#plt.xlabel('Simulation run')
#plt.ylabel('QoE sum')
#plt.show()
print(simulator.abr_controller.eps, simulator.speed_controller.eps)
