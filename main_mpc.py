import Simulator
import AbrController
import SpeedController
import HeuristicSpeedController
import DQNSpeedController
import mpc

# constants
INTERVAL = 0.5  # interval of network trace
CHUNK_LEN = 1   # default length for a chunk
MAX_BUFFER = 5
START_UP = 2

BITR_W = 1
REBUF_W = 1
VAR_W = 1
START_W = 1
LAT_W = 1

## DO NOT MODIFY THIS PART
# simulator initiation
simulator = Simulator.Simulator()

mpdfile = "trace/example_video"
networktrace = "trace/example_networktrace"
simulator.set_mpd(CHUNK_LEN, MAX_BUFFER, START_UP, mpdfile)
simulator.set_network_info(INTERVAL, networktrace)

qoe_metric = Simulator.QOEMetric(BITR_W, REBUF_W, VAR_W, START_W, LAT_W)
simulator.set_qoe_metric(qoe_metric)

# controller initiation
abr_controller = mpc.MPCBitrateController(simulator)
speed_controller = DQNSpeedController.SpeedController(simulator)

# set controller
simulator.abr_controller = abr_controller
simulator.speed_controller = speed_controller

## DO NOT MODIFY THIS PART

qoe = simulator.run()
print(qoe)
