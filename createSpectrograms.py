import os
import subprocess
import argparse
from tqdm import tqdm  #For progress bars etc

parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir', default='data/wav_songs', help="Directory with the .wav SONGS dataset")
parser.add_argument('--output_dir', default='data/raw_specs', help="Where to write the new data")
parser.add_argument('--pixpersec', default= 25, help="Number of pixels per second in the spectrogram")

if __name__ == '__main__':
	args = parser.parse_args()

	audio_dir = args.audio_dir   # source of the .wav songs
	specs_dir = args.output_dir  # Output directory for the raw spectrograms
	pixpersec = args.pixpersec   # number of pixels per second in the spectrogram

	assert os.path.isdir(audio_dir), "Couldn't find the dataset at {}".format(audio_dir)

	if not os.path.exists(specs_dir):
		os.mkdir(specs_dir)
	else:
		print("Warning: output dir {} already exists".format(specs_dir))


	fnames = os.listdir(audio_dir)
	fnames = [os.path.join(audio_dir, f) for f in fnames if f.endswith(".wav")]

	for fname in tqdm(fnames):
		dest = os.path.join(specs_dir, str(fname.split("/")[-1].split(".")[0])+".png")
		"""
		Create a command for the os to run sox and create a spectrogram
		remix 1 selects the left channel of the music
		-Y 200 sets the y axis to be 200 pixels
		-X pixpersec sets the number of pixels per second in the spectrogram
		-m for grayscale
		-r Get a raw spectrogram without axes
		-o output file
		"""

		command = ['sox', fname, '-n', 'remix', '1','spectrogram', '-Y', '200', '-X', str(pixpersec), '-m', '-r', '-o', dest]
		subprocess.call(command)






