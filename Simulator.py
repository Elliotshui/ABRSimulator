import Plot
import matplotlib.pyplot as plt
# bitrates is the list of bitrate available for a chunk
# normally four or five options, bitrates[0] stands for the lowest
class Chunk(object):
    def __init__(self, bitrates):
        self.bitrates = bitrates

# video_length: the number of chunks
# chunk_length: the default length for every chunk
# max_buffer: maximum number of chunks in the buffer
class MPD(object):
    def __init__(self, video_length, chunk_length, max_buffer, start_up_length, chunks):
        self.video_length = video_length
        self.chunk_length = chunk_length
        self.max_buffer = max_buffer
        self.start_up_length = start_up_length
        self.chunks = chunks

class QOEMetric(object):
    def __init__(self, bitrate_weight, rebuffer_weight, variance_weight, startup_weight, latency_weight):
        self.bitrate_weight = bitrate_weight
        self.rebuffer_weight = rebuffer_weight
        self.variance_weight = variance_weight
        self.startup_weight = startup_weight
        self.latency_weight = latency_weight

# chunk_id: the id of the chunk to be downloaded
# previous_bitrates: the bitrate of the previous chunk, for calculating the variance
# previous_bandwidths: the list of bandwidths, for predicting the future bandwidth
# buffer_level: current buffer length (in seconds)
class ChunkInfo(object):
    def __init__(self, chunk_id, previous_bitrates, previous_bandwidths, buffer_level):
        self.chunk_id = chunk_id
        self.previous_bitrates = previous_bitrates
        self.previous_bandwidths = previous_bandwidths
        self.buffer_level = buffer_level

# Assume the throughput of network is a square wave function
# bandwidths[i] stands for the bandwidth in the ith interval
class NetworkInfo(object):
    def __init__(self, interval, bandwidths):
        self.interval = interval
        self.bandwidths = bandwidths


class Simulator(object):
    def __init__(self):
        self.qoe_metric = None
        self.mpd = None
        self.network_info = None
        self.abr_controller = None
        self.speed_controller = None

    def set_qoe_metric(self, qoe_metric):
        self.qoe_metric = qoe_metric
        return
    
    # read network info from net work trace file
    def set_network_info(self, interval, networktrace):
        f = open(networktrace)
        bandwidths = []
        for line in f.readlines():
            bandwidths.append(float(line))
        self.network_info = NetworkInfo(interval, bandwidths)
        return 

    # read mpd from mpd file 
    def set_mpd(self, chunk_length, max_buffer, start_up_length, mpdfile):
        f = open(mpdfile)
        video_length = 0
        chunks = []
        for line in f.readlines():
            video_length += 1
            bitrates = []
            for item in line.split():
                bitrates.append(float(item))
            chunks.append(Chunk(bitrates))
        self.mpd = MPD(video_length, chunk_length, max_buffer, start_up_length, chunks)
        return

    def calculate_qoe(self, rebuffer_time, previous_bitrates, start_up_time, average_latency):
        variance = 0
        for i in range(0, self.mpd.video_length - 1):
            variance += abs(self.mpd.chunks[i].bitrates[previous_bitrates[i]] - self.mpd.chunks[i + 1].bitrates[previous_bitrates[i + 1]])

        total_bitrate = 0
        for i in range(0, self.mpd.video_length):
            total_bitrate += self.mpd.chunks[i].bitrates[previous_bitrates[i]]

        return self.qoe_metric.bitrate_weight * total_bitrate, \
               self.qoe_metric.rebuffer_weight * rebuffer_time, \
               self.qoe_metric.variance_weight * variance, \
               self.qoe_metric.startup_weight * start_up_time, \
               self.qoe_metric.latency_weight * average_latency
    
    def get_mpd(self):
        return self.mpd

    def get_qoe_metric(self):
        return self.qoe_metric


    # run the simulation for the whole process
    def run(self):
        
        # download state
        chunk_id = 0                # chunk_id to be downloaded
        available_id = -1           # the lastest available chunk_id
        previous_bitrates = []      # bitrate index for previous chunks
        current_bitrate = None      # bitrate index for the chunk to be downloaded
        previous_bandwidths = []    # list of previous average bandwidths
        previous_download_times = []      
        downloaded_size = 0.0       
        target_size = None
        download_pause = True
        download_time = 0
        
        # buffer state
        buffer_level = 0            
        max_buffer = self.mpd.max_buffer
        buffer_empty = True
        buffer_full = False

        # playback state
        play_id = 0                 # id for current playing chunk
        play_length = 0             # play length for current playing chunk
        play_time = 0               # total play time of the video, used to calculate the latency
        play_speed = None
        play_pause = True 

        #latency state
        instant_latency = 0
        average_latency = 0

        # simulation state
        start_up = True
        simulation_end = False
        
        # timer
        global_time = 0.0
        rebuffer_time = 0.0
        last_rebuffer_time = 0.0
        start_up_time = 0.0

        # update the state for every dt second, loop until the end
        dt = 0.01

        # list of info for plotting
        time_list = []
        buffer_list = []
        rebuffer_list = []
        latency_list = []

        while simulation_end == False:
            # insert plot info
            time_list.append(global_time)
            buffer_list.append(buffer_level)
            rebuffer_list.append(rebuffer_time)
            latency_list.append(instant_latency)
            
            # update timers
            if start_up == True:
                start_up_time += dt
            elif buffer_empty == True:
                rebuffer_time += dt

            # if the download pauses
            available_id = min(int(global_time / self.mpd.chunk_length) - 1, self.mpd.video_length - 1)
            if available_id < chunk_id or buffer_full == True:
                download_pause = True
            else:
                download_pause = False
            
            # if the playback pauses
            if buffer_empty == True or start_up == True:
                play_pause = True
            else:
                play_pause = False

            # if downloading, download the video and update the state
            if download_pause == False:                
                # if downloading a new chunk, call the abr controller to determine the bitrate
                if download_time == 0:
                    rebuf_for_chunk = rebuffer_time - last_rebuffer_time
                    current_bitrate = self.abr_controller.get_next_bitrate(chunk_id, previous_bitrates, previous_bandwidths, previous_download_times, buffer_level, rebuf_for_chunk)
                    last_rebuffer_time = rebuffer_time
                    target_size = self.mpd.chunks[chunk_id].bitrates[current_bitrate] * self.mpd.chunk_length             
                # calculate the instant bandwidth
                bandwidth_idx = int(global_time / self.network_info.interval)
                bandwidth = self.network_info.bandwidths[bandwidth_idx]
                downloaded_size = downloaded_size + bandwidth * dt
                download_time += dt
                #if finished downloading the current chunk, update the chunk to be downloaded
                if downloaded_size >= target_size:
                    previous_bandwidths.append(downloaded_size / download_time)
                    previous_bitrates.append(current_bitrate)
                    previous_download_times.append(download_time)
                    chunk_id += 1
                    downloaded_size = 0
                    download_time = 0
                    # only update buffer when the whole chunk is downloaded
                    buffer_level += self.mpd.chunk_length

            # calculate instant latency
            instant_latency = global_time - play_time
            average_latency = (average_latency * global_time + instant_latency * dt) / (global_time + dt)    
            
            # if playing, consume the buffer and update the state
            if play_pause ==  False:
                # at start of each chunk, determine the speed
                if play_length == 0:
                    play_speed = self.speed_controller.get_next_speed()
                # update the playback state
                play_time += play_speed * dt    
                play_length += play_speed * dt
                buffer_level -= play_speed * dt
                if play_length >= self.mpd.chunk_length:
                    play_length = 0
                    play_id += 1
            
            # update buffer state
            if buffer_level >= max_buffer:
                buffer_full = True
            else:
                buffer_full = False
            if buffer_level <= 0:
                buffer_level = 0
                buffer_empty = True
            else:
                buffer_empty = False
            
            # update start up state
            if start_up == True and buffer_level >= self.mpd.start_up_length:
                start_up = False

            # update global timer
            global_time += dt

            if chunk_id >= self.mpd.video_length and play_id >= self.mpd.video_length:
                simulation_end = True

        Plot.Plot(time_list, buffer_list, "buffer level").plot_info()
        Plot.Plot(time_list, rebuffer_list, "rebuffer time").plot_info()
        Plot.Plot(time_list, latency_list, "instant latency").plot_info()
        plt.show()

        return self.calculate_qoe(rebuffer_time, previous_bitrates, start_up_time, average_latency)