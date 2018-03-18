import os
import argparse
import random
from shutil import copyfile
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/fma_small', help="Directory with all songs")
parser.add_argument('--num_select', default= 1000, help="Number of songs to select for train/dev/test")
parser.add_argument('--output_dir', default='data/raw_songs', help="Where to put the songs to be used")


if __name__ == '__main__':
	args = parser.parse_args()

	all_songs = args.data_dir
	raw_songs = args.output_dir
	num_select = args.num_select

	assert os.path.isdir(all_songs), "Couldn't find the dataset at {}".format(all_songs)
	fnames = os.listdir(all_songs)
	fnames = [os.path.join(all_songs, f) for f in fnames if f.endswith(".mp3")]

	#Split the files into the different sets. Use seed to get consistent results on all runs
	random.seed(230)
	fnames.sort()
	random.shuffle(fnames)

	split = num_select if num_select <= len(fnames) else len(fnames) 
	sel_fnames = fnames[:split]
	
	if not os.path.exists(raw_songs):
		os.mkdir(raw_songs)
	else:
		print("Warning: output dir {} already exists".format(raw_songs))

	for f in tqdm(sel_fnames):
		dst = os.path.join(raw_songs, f.split("/")[-1])

		copyfile(f, dst)

	print("Phew! I am finally done building the dataset!")

