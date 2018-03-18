import os
import subprocess
import IPython.display as ipd
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mp3_dir', default='data/classified_songs', help="Directory with the raw SONGS dataset")
parser.add_argument('--wav_dir', default='data/wav_songs', help="Where to write the new data")

if __name__ == '__main__':
	args = parser.parse_args()

	assert os.path.isdir(args.mp3_dir), "Couldn't find the dataset at {}".format(args.mp3_dir)

	mp3_dir = args.mp3_dir
	wav_dir = args.wav_dir

	if not os.path.exists(wav_dir):
		os.mkdir(wav_dir)
	else:
		print("Warning: output dir {} already exists".format(wav_dir))

	for fname in tqdm(os.listdir(args.mp3_dir)):
		if fname.endswith(".mp3"):
			command = ['ffmpeg', '-v', '0', '-i', os.path.join(mp3_dir, fname), os.path.join(wav_dir, str(fname.split(".")[0])+ ".wav")]
			subprocess.call(command)
