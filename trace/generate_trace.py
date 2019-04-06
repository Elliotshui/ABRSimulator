import numpy as np

# generate video trace
video_length = 60
bitrates = [240, 320, 480, 720, 1080]
f_video = open("example_video", "r+")
for i in range(0, video_length):
    for j in range(0, 5):
        f_video.write(str(bitrates[j]) + ' ')
    f_video.write('\n')

# generate network trace
network_length = 100000
f_net = open("example_networktrace", "r+")

average_bandwidth = 500
sigma = 100
for i in range(0, network_length):
    rand_bandwidth = np.random.randn() * sigma + average_bandwidth
    if(rand_bandwidth < 0):
        rand_bandwidth = 0
    f_net.write(str(rand_bandwidth) + '\n')