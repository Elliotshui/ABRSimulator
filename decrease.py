import os
import numpy as np


IN_FILE = '/Users/cici/Desktop/cooked/'
OUT_FILE = '/Users/cici/Desktop/cooked2/'


def main():
	files = os.listdir(IN_FILE)
	for trace_file in files:
			with open(IN_FILE + trace_file, 'rb') as f, open(OUT_FILE + trace_file, 'wb') as mf:
				for line in f:
					throughput = float(line.split()[0])
					throughput = throughput/4500

					mf.write(str(throughput) + '\n')


if __name__ == '__main__':
	main()
