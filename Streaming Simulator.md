## Streaming Simulator

Since the target of our work is to evaluate different algorithms that optimize the overall QoE in HTTP video live-streaming scenario, and some of them needs to use reinforcement learning methods, we developed a faithful DASH streaming simulator which accelerates the training speed compared to training in real environment. By using the simulator, the RL algorithm doesn't have to wait until all of the video are downloaded before updating the model. Also, the simulator measures the QoE for a given algorithm, which is convenient for testing.   

The simulator is composed of three parts:  a live-stream publisher that continuously generates video contents,  a client that receive the contents, and an ABRcontroller that choose the bitrate for future video chunks.  As is mentioned in previous chapter, the client also has a speed controller which can adjust playback speed at the beginning of each chunk. It works independently from the ABRcontroller.

For simulating the live-stream publisher, rather than generate the video chunks during the simulation, we have the all the video chunks with different bitrates beforehand,  and marks some of them as available during the process. The client cannot download a chunk that has not been marked as available, which is a major difference between live-streaming and VOD. If the bandwidth is really high and the previous chunk is completely downloaded before the next chunk is generated,  then the client have to pause the download and wait for the next chunk.

The simulator represents the clients' playback buffer as a float number, which stands for how many seconds of video have already been downloaded and decoded.  During the download of a video chunk, the  buffer does not increase because the decode requires the whole chunk to be downloaded. Only after a whole chunk is downloaded and decoded, the buffer level will increase. We ignore the time for decoding because it's rather short and irrelevant to the network condition. 

When the buffer is empty, the play back is paused and the simulator records the rebuffer time until the next chunk in the buffer.  When the buffer level exceeds the maximum value (usually specified in the mpd file), the download for the next chunk will be paused until the buffer level is below the maximum.

At the beginning of each chunk download, the client turns to the ABRcontroller to choose the best bitrate given the current state. For every chunk to be played, the client uses the speed controller to choose the play back speed. Though the ABRcontroller is our major focus, how to design a speed conroller that performs well is also a challenge, our simluator only implements a simple one based on buffer level.

 



