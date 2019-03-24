import os, sys, random

# 68 files with 2000 seconds, 205 files with 320 seconds

os.system("mkdir synthetic_traces_complete")
os.system("mkdir synthetic_traces_complete\\training")
os.system("mkdir synthetic_traces_complete\\testing")

#generate 66 files, each 2000 seconds for training
#for i in range(0, 5):
#    print(i)
#    name = "synthetic_traces_complete/training/trace" + str(i) + ".txt"
#    os.system("python pensieve_synthetic_trace.py " + str(random.uniform(10, 100)) + " " + str(random.uniform(1,5)) + " " + str(random.uniform(0.05, 0.5)) + " 2000 > " + name)

# generate 205 files, each 320 seconds for testing
#for i in range(0, 205):
#    name = "synthetic_traces_complete/testing/trace" + str(i) + ".txt"
#    os.system("python pensieve_synthetic_trace.py " + str(random.uniform(10, 100)) + " " + str(random.uniform(1,5)) + " " + str(random.uniform(0.05, 0.5)) + " 320 > " + name)

# generate N files, each 60 seconds for training
N = 500
for i in range(0, N):
    print(i/N)
    name = "synthetic_traces_complete/training/trace" + str(i) + ".txt"
    os.system("python pensieve_synthetic_trace.py " + str(random.uniform(10, 100)) + " " + str(random.uniform(1,5)) + " " + str(random.uniform(0.05, 0.5)) + " 60 > " + name)
