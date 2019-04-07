import Simulator
from Simulator import MPD, QOEMetric, NetworkInfo, Chunk
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


class Benchmarker:
    def __init__(self):
        pass

    def run_single_tests(self, controllers_list, test_params):
        test_list = []
        for abr_controller, speed_controller in controllers_list:
            test_name = (abr_controller.name + ' + ' +speed_controller.name
                         + ':  ' + test_params.name)
            test = self.setup_test(test_name, abr_controller,
                       speed_controller, test_params)
            test.run_test()
            test_list.append(test)
        return test_list

    def print_single_tests(self, test_list):
        print('____Results of evaluation on a single trace and mpd____')
        for test in test_list:
            print(test)

    def plot_single_tests(self, test_list):
        plot_height = len(test_list)
        plot_width = 2
        position = 1
        plt.figure()
        for test in test_list:
            plt.subplot(plot_height, plot_width, position)
            position += 1
            plt.plot(test.chunk_history[:-1], test.speed_history)
            plt.title("Speed vs. chunk id")

            plt.subplot(plot_height, plot_width, position)
            position += 1
            plt.plot(test.chunk_history, test.bitrate_history)
            plt.plot(test.chunk_history, test.bandwidth_history)
            plt.title("Bitrate and bandwidth vs. chunk id")

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                            top=0.9, wspace=0.45, hspace=0.95)
        plt.show()

    def run_batch_tests(self, controllers_list, test_params_list):
        test_superlist = []
        for test_params in test_params_list:
            test_list = []
            for abr_controller, speed_controller in controllers_list:
                test_name = (abr_controller.name + ' + ' +speed_controller.name
                             + ':  ' + test_params.name)
                test = self.setup_test(test_name, abr_controller,
                           speed_controller, test_params)
                test.run_test()
                test_list.append(test)
            test_superlist.append(test_list)
        return test_superlist

    def print_batch_tests(self, test_superlist):
        first_test = test_superlist[0]
        total_qoe_sums = np.zeros((len(first_test)))
        for test_list in test_superlist:
            for i, test in enumerate(test_list):
                total_qoe_sums[i] += test.final_qoe_sum

        print("____Results of evaluation on multiple traces and mpds____")
        for i, sum in enumerate(total_qoe_sums):
            print(first_test[i].simulator.abr_controller.name + ' + '
                  + first_test[i].simulator.speed_controller.name + ' total: '
                  + str(sum))

    def setup_test(self, test_name, abr_controller,
                   speed_controller, test_params):
        test = Test(test_name)
        simulator = Simulator.Simulator()
        simulator.auto_extend_trace = False
        simulator.display_plots = False
        simulator.qoe_metric = test_params.qoe_metric
        simulator.mpd = test_params.mpd
        simulator.network_info = test_params.network_info

        abr_controller.assign_simulator(simulator)
        simulator.abr_controller = abr_controller
        speed_controller.assign_simulator(simulator)
        simulator.speed_controller = speed_controller

        test.simulator = simulator
        test.test_params = test.test_params
        return test

    def new_test_params(self, name):
        return TestParams(name)

    def load_trace_from_file(self, file_path):
        f = open(file_path)
        bandwidths = []
        for line in f.readlines():
            bandwidths.append(float(line))
        return bandwidths

    def load_chunks_from_file(self, file_path):
        f = open(file_path)
        chunks = []
        for line in f.readlines():
            bitrates = []
            for item in line.split():
                bitrates.append(float(item))
            chunks.append(Chunk(bitrates))
        return chunks

    def dummy_simulator(self):
        dummy = Simulator.Simulator()
        bitrates = (1, 1, 1, 1, 1)
        chunks = [Chunk(bitrates)]
        network_trace = []
        dummy_params = self.new_test_params('')
        qoe_metric = dummy_params.set_qoe_metric(
            bitrate_weight = 1,
            rebuffer_weight = 1000,
            variance_weight = 1,
            startup_weight = 1000,
            latency_weight = 0)
        mpd = dummy_params.set_mpd(
            chunk_length = 1,
            max_buffer = 15,
            start_up_length = 2,
            chunks = chunks)
        network_info = dummy_params.set_network_info(
            trace_interval = 0.5,
            network_trace = network_trace)
        dummy.qoe_metric = qoe_metric
        dummy.mpd = mpd
        dummy.network_trace = network_trace
        return dummy


class Test:
    def __init__(self, test_name, simulator=None, test_params=None):
        self.test_name = test_name
        self.test_simulator = simulator
        self.test_params = test_params
        self.finished = False

    def __str__(self):
        s = ''
        if self.finished:
            s += '---' + self.test_name +'---'
            s += '\nFinal qoe sum: {}'.format(self.final_qoe_sum)
            s += '\nFinal qoe terms: {}'.format(self.final_qoe_terms)
            s += '\nAverage qoe: {}'.format(self.average_qoe)
            s += '\nAverage bandwidth: {}'.format(self.average_bandwidth)
        else:
            s += '---' + self.test_name +'---\n'
            s += 'Test not run'
        return s

    def run_test(self):
        simulator = self.simulator
        self.final_qoe_terms = simulator.run()
        self.final_qoe_sum = self.final_qoe_terms[0] - sum(self.final_qoe_terms[1:])

        self.chunk_history = simulator.chunk_history
        self.bitrate_history = simulator.bitrate_history
        self.bandwidth_history = simulator.bandwidth_history
        self.speed_history = simulator.speed_history

        self.time_list = simulator.time_list
        self.buffer_list = simulator.buffer_list
        self.rebuffer_list = simulator.rebuffer_list
        self.latency_list = simulator.latency_list
        self.qoe_list = simulator.qoe_list

        self.average_bitrate = self.average(self.bitrate_history)
        self.average_speed = self.average(self.speed_history)
        self.average_bandwidth = self.average(self.bandwidth_history)
        self.average_qoe = self.average(self.qoe_list)
        self.finished = True

    def average(self, value_list):
        return sum(value_list)/len(value_list)

    def display_time_plots(self):
        plt.subplot(1, 3, 1)
        plt.plot(self.time_list, self.buffer_list)
        plt.title('Buffer level')
        plt.subplot(1, 3, 2)
        plt.plot(self.time_list, self.rebuffer_list)
        plt.title('Rebuffer timer')
        plt.subplot(1, 3, 3)
        plt.plot(self.time_list, self.latency_list)
        plt.title('Latency')
        plt.show()

    def display_chunk_plots(self):
        pass


class TestParams:
    def __init__(self, name, qoe_metric=None, mpd=None, network_info=None):
        self.name = name
        self.qoe_metric = qoe_metric
        self.mpd = mpd
        self.network_info = network_info

    def set_qoe_metric(self, bitrate_weight, rebuffer_weight, variance_weight,
                       startup_weight, latency_weight):
        self.qoe_metric = QOEMetric(
            bitrate_weight, rebuffer_weight, variance_weight,
            startup_weight, latency_weight)
        return self.qoe_metric

    def set_mpd(self, chunk_length, max_buffer, start_up_length, chunks):
        video_length = len(chunks)
        self.mpd = MPD(
            video_length, chunk_length, max_buffer, start_up_length, chunks)
        return self.mpd

    def set_network_info(self, trace_interval, network_trace):
        self.network_info = NetworkInfo(trace_interval, network_trace)
        return self.network_info


def test_single_benchmarking():
    # This code illustrates how to test multiple controllers on a single
    # network trace and mpd pair.
    # Init benchmarker and choose network trace and chunk files
    bm = Benchmarker()
    network_trace = bm.load_trace_from_file(r'benchmark_traces\trace1_13008_cnn.txt')
    chunks = bm.load_chunks_from_file(r'benchmark_chunks\chunks1.txt')
    # Set up testing parameters
    params = bm.new_test_params('Hello world')
    params.set_qoe_metric(
        bitrate_weight = 1,
        rebuffer_weight = 1000,
        variance_weight = 1,
        startup_weight = 1000,
        latency_weight = 0)
    params.set_mpd(
        chunk_length = 1,
        max_buffer = 15,
        start_up_length = 2,
        chunks = chunks)
    params.set_network_info(
        trace_interval = 0.5,
        network_trace = network_trace)
    # Choose controllers. Controller should already be initialized, although
    # in this case I use new ones initalized with a dummy simulator.
    # Controllers must have method .assign_simulator(simulator) that takes a
    # simulator object and updates the controllers mpd and qoe_metric.
    # Controllers must also have a .name attribute containing a string
    controllers1 = (mpc_abr.BitrateController(bm.dummy_simulator()),
                    HeuristicSpeedController.SpeedController(bm.dummy_simulator()))
    controllers2 = (bola_abr.BitrateController(bm.dummy_simulator()),
                    HeuristicSpeedController.SpeedController(bm.dummy_simulator()))
    # Form a list of all controllers to be tested
    controllers_list = [controllers1, controllers2]
    # Run tests. test_list is a list of test objects containing the results
    test_list = bm.run_single_tests(controllers_list, params)
    # Visualize through printing
    bm.print_single_tests(test_list)
    # Visualize through plotting
    bm.plot_single_tests(test_list)
    # It is also possible to plot and print individual tests
    print(test_list[0])
    test_list[0].display_time_plots()

def test_batch_benchmarking():
    # Works similar to single benchmark, but uses a list of test_param objects
    bm = Benchmarker()

    params1 = bm.new_test_params('CNN Balanced')
    network_trace1 = bm.load_trace_from_file(r'benchmark_traces\trace1_13008_cnn.txt')
    chunks1 = bm.load_chunks_from_file(r'benchmark_chunks\chunks1.txt')
    params1.set_qoe_metric(
        bitrate_weight = 1,
        rebuffer_weight = 1000,
        variance_weight = 1,
        startup_weight = 1000,
        latency_weight = 0)
    params1.set_mpd(
        chunk_length = 1,
        max_buffer = 15,
        start_up_length = 2,
        chunks = chunks1)
    params1.set_network_info(
        trace_interval = 0.5,
        network_trace = network_trace1)

    params2 = bm.new_test_params('Ebay Balanced')
    network_trace2 = bm.load_trace_from_file(r'benchmark_traces\trace2_12912_ebay.txt')
    chunks2 = bm.load_chunks_from_file(r'benchmark_chunks\chunks1.txt')
    params2.set_qoe_metric(
        bitrate_weight = 1,
        rebuffer_weight = 1000,
        variance_weight = 1,
        startup_weight = 1000,
        latency_weight = 0)
    params2.set_mpd(
        chunk_length = 1,
        max_buffer = 15,
        start_up_length = 2,
        chunks = chunks2)
    params2.set_network_info(
        trace_interval = 0.5,
        network_trace = network_trace2)

    controllers1 = (mpc_abr.BitrateController(bm.dummy_simulator()),
                    HeuristicSpeedController.SpeedController(bm.dummy_simulator()))
    controllers2 = (bola_abr.BitrateController(bm.dummy_simulator()),
                    HeuristicSpeedController.SpeedController(bm.dummy_simulator()))

    params_list = [params1, params2]
    controllers_list = [controllers1, controllers2]
    # test_superlist is a list of test_lists's
    test_superlist = bm.run_batch_tests(controllers_list, params_list)
    bm.print_batch_tests(test_superlist)

def test_dqn_benchmarking():
    # This code illustrates how to test multiple controllers on a single
    # network trace and mpd pair.
    # Init benchmarker and choose network trace and chunk files
    bm = Benchmarker()
    network_trace = bm.load_trace_from_file(r'benchmark_traces\trace1_13008_cnn.txt')
    chunks = bm.load_chunks_from_file(r'benchmark_chunks\chunks1.txt')
    # Set up testing parameters
    params = bm.new_test_params('Hello world')
    params.set_qoe_metric(
        bitrate_weight = 1,
        rebuffer_weight = 1000,
        variance_weight = 1,
        startup_weight = 1000,
        latency_weight = 0)
    params.set_mpd(
        chunk_length = 1,
        max_buffer = 15,
        start_up_length = 2,
        chunks = chunks)
    params.set_network_info(
        trace_interval = 0.5,
        network_trace = network_trace)
    # Choose controllers. Controller should already be initialized, although
    # in this case I use new ones initalized with a dummy simulator.
    # Controllers must have method .assign_simulator(simulator) that takes a
    # simulator object and updates the controllers mpd and qoe_metric.
    # Controllers must also have a .name attribute containing a string
    dqn_bitrate_controller = DQNBitrateController.BitrateController(bm.dummy_simulator())
    dqn_bitrate_controller.load_model('saved_models\\training_run_2\\51853abr_5.ckpt')
    controllers1 = (dqn_bitrate_controller,
                    HeuristicSpeedController.SpeedController(bm.dummy_simulator()))

    controllers2 = (bola_abr.BitrateController(bm.dummy_simulator()),
                    HeuristicSpeedController.SpeedController(bm.dummy_simulator()))
    # Form a list of all controllers to be tested
    controllers_list = [controllers1, controllers2]
    # Run tests. test_list is a list of test objects containing the results
    test_list = bm.run_single_tests(controllers_list, params)
    # Visualize through printing
    bm.print_single_tests(test_list)
    # Visualize through plotting
    bm.plot_single_tests(test_list)


#test_single_benchmarking()
#test_batch_benchmarking()
test_dqn_benchmarking()
