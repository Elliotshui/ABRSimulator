import random
import numpy as np
import matplotlib.pyplot as plt # disable in order to run on server
import time
from scipy.signal import savgol_filter
import os

MIN_REAL_AVG = 0.2
MAX_REAL_AVG = 4.0

class TraceFetcher:
    def __init__(self, trace_start_number=0,
                 real_traces=False, trace_folder_path=None,
                 dt=None, maximum_bw=None, minimum_bw=None,
                 fine_variability=None, course_variability=None,
                 course_freq=None, smoothing_factor=None, post_noise=None):
        self.trace_number = trace_start_number

        self.real_traces = real_traces
        self.trace_folder_path = trace_folder_path

        self.dt = dt
        self.maximum_bw = maximum_bw
        self.minimum_bw = minimum_bw
        self.fine_variability = fine_variability
        self.course_variability = course_variability
        self.course_freq = course_freq
        self.smoothing_factor = smoothing_factor
        self.post_noise = post_noise


    def get_random_trace(self, trace_length):
        if self.real_traces:
            trace = self.get_random_real_trace(
                trace_length, self.trace_folder_path)
        else:
            timesteps, trace = self.generate_trace(
                self.dt,
                trace_length,
                self.maximum_bw,
                self.minimum_bw,
                self.fine_variability,
                self.course_variability,
                self.course_freq,
                self.post_noise)

        return trace

    def bound(self, maximum_bw, minimum_bw, bw):
        if bw > maximum_bw:
            return maximum_bw
        elif bw < minimum_bw:
            return minimum_bw
        else:
            return bw

    def generate_trace(self, dt, length, maximum_bw, minimum_bw, fine_variability,
                       course_variability, course_freq, post_noise,
                       smoothing_factor=0):
        """Generate network trace and return timesteps and bandwidths arrays.
        Arguments:
            dt: float of timestep in seconds
            length: float of length of trace in seconds
            maximum_bw: int of maximum allowed bandwidth in kbps
            minimum_bw: int of maximum allowed bandwidth in kbps
            fine_variability: float controlling fine variations in bw,
                              recommended values between 0 and 1
            course_variability: float controlling course variations in bw,
                                recommended values between 0 and 1
            course_freq: int of avg seconds between course variability changes
            smoothing_factor: float between 0 and 1, strength of smoothing
            """
        # Recalculate course_freq in terms of time steps
        course_freq = int(course_freq/dt)
        # Calculate number of time steps
        num_steps = int(np.ceil(length/dt))+1
        # Generate fine throughput variations
        fine_std = 25 * fine_variability
        fine_steps = np.random.randn((num_steps)) * fine_std
        # Generate course throughput variations
        course_std = np.sqrt((maximum_bw - minimum_bw) / 2) * 10 * course_variability
        course_steps = np.random.randn((num_steps)) * course_std
        # Initialize arrays
        timesteps = np.zeros((num_steps))
        timesteps[0] = 0
        bandwidths = np.zeros((num_steps))
        bw0 = np.random.rand() * (maximum_bw - minimum_bw) + minimum_bw
        bandwidths[0] = bw0
        # Generate throughput trace
        t = 0
        i = 0
        course_count = 1
        next_course = course_freq
        while i < num_steps-1:
            if course_count % next_course == 0:
                new_bw = bandwidths[i] + fine_steps[i] + course_steps[i]
                course_count = 1
                # Choose number of steps until next course with some variation
                course_rnd_shift = np.random.randint(-course_freq//2,
                                                     course_freq//2)
                next_course = course_freq + course_rnd_shift
            else:
                new_bw = bandwidths[i] + fine_steps[i]
                course_count += 1
            bandwidths[i+1] = self.bound(maximum_bw, minimum_bw, new_bw)
            timesteps[i+1] = t
            t += dt
            i += 1

        if smoothing_factor:
            # Ensure window length is odd and larger than polyorder
            window_length = int(((smoothing_factor * num_steps)//2)*2 + 1)
            polyorder = 1
            if window_length <= polyorder:
                window_length = polyorder+1 if polyorder%2 else polyorder+2
            # Apply smoothing
            bandwidths = savgol_filter(bandwidths, window_length, polyorder)

        noise = np.random.randn((num_steps)) * post_noise
        bandwidths += noise
        return timesteps, bandwidths

    def import_trace(self, trace_path):
        f = open(trace_path)
        bandwidths = []
        for line in f.readlines():
            bandwidths.append(float(line)/1000)
        return bandwidths

    def get_random_real_trace(self, trace_length, trace_folder_path):
        list_of_trace_names = os.listdir(trace_folder_path)
        avg_acceptable = False
        while not avg_acceptable:
            new_trace = []
            while len(new_trace) < trace_length:
                trace_name = random.choice(list_of_trace_names)
                trace_path = trace_folder_path + "\\" + trace_name
                new_trace += self.import_trace(trace_path)
            if len(new_trace) > trace_length:
                new_trace = new_trace[:trace_length]
            trace_avg = sum(new_trace)/len(new_trace)
            if trace_avg > MIN_REAL_AVG and trace_avg < MAX_REAL_AVG:
                avg_acceptable = True
        return new_trace

    def plot_trace(self, timesteps, bandwidths):
        plt.plot(timesteps, bandwidths)
        plt.title('Network trace')
        plt.xlabel('Time [s]')
        plt.ylabel('Throughput [kbps]')
        plt.show()

    def plot_random_trace(self):
        trace = self.get_random_real_trace(60, "C:\\Users\\Einar Lenneloev\\Desktop\\Traces\\cooked3")

        self.plot_trace(range(len(trace)), trace)

def time_test():
    generator = TraceFetcher(0)
    dt = 1
    length = 60
    maximum_bw = 6000
    minimum_bw = 200
    fine_variability = 0.1
    course_variability = 0.25
    course_freq = 10
    post_noise = 5
    smoothing_factor = 0.1

    trace_list = []
    start = time.time()
    num_traces = 6000
    for i in range(0, num_traces):
        timesteps, bandwidths = generator.generate_trace(
            dt, length, maximum_bw, minimum_bw, fine_variability,
            course_variability, course_freq, post_noise, smoothing_factor)
        trace_list.append((timesteps, bandwidths))
    end = time.time()
    print('Time to generate {} traces of length {} seconds: {}s'.format(
        num_traces, length, end - start))


def plot_test():
    generator = TraceFetcher(0)
    dt = 0.1  # Timestep in seconds
    length = 60  # Length of trace in seconds
    maximum_bw = 6000  # Maximum allowed bandwidth
    minimum_bw = 200  # Minimum allowed bandwidth
    fine_variability = 0.2  # Control fine variation amplitude
    course_variability = 0.5  # Control course variation amplitude
    course_freq = 10  # Control course variation frequency
    smoothing_factor = 0.1 # Control strength of smoothing
    post_noise = 10  # Standard deviation of post-noise

    timesteps, bandwidths = generator.generate_trace(
        dt, length, maximum_bw, minimum_bw, fine_variability,
        course_variability, course_freq, post_noise, smoothing_factor)
    generator.plot_trace(timesteps, bandwidths)

if __name__ == "__main__":
    #time_test()
    #plot_test()
    trf = TraceFetcher()
    trf.plot_random_trace()
