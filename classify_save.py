"""
Classify_save takes songs from the fma_small database and uses the tracks.csv
file from the fma_metadata to classify the songs. It adds a number representing the song
genre before the number of the track.
The following encoding is used:

	0 - Electronic
	1 - Experimental
	2 - Folk
	3 - Hip-Hop
	4 - Instrumental
	5 - International
	6 - Pop
	7 - Rock
"""

import os
import subprocess
import IPython.display as ipd
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/raw_songs', help="Directory with the raw SONGS dataset")
parser.add_argument('--meta_data', default='data/tracks.csv', help="csv file with song meta_data")
parser.add_argument('--output_dir', default='data/classified_songs', help="Where to write the new data")


def rename_and_save(data_dir, output_dir, song_dict, genre_dict):
	fnames = os.listdir(args.data_dir);

	for filename in tqdm(fnames):
		if filename.endswith(".mp3"):
			temp = filename
			lhs, _ = filename.split(".")
			song_type = song_dict[int(lhs)]
			type_code = genre_dict[str(song_type)]
			os.rename(os.path.join(data_dir, filename), os.path.join(output_dir, str(type_code)+"_"+temp))

if __name__ == '__main__':
	args = parser.parse_args()

	assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

	df = pd.read_csv(args.meta_data, skiprows  = 1); 
	df2 = pd.read_csv(args.meta_data, skiprows  = 2);

	df = df.loc[1:, "genre_top" ];
	df2 = df2.loc[:, "track_id"];

	genre_list = df.values.T.tolist();
	song_id = df2.values.T.tolist();

	genre_dict = {"Electronic" :0, "Experimental" :1, "Folk" :2, "Hip-Hop":3, "Instrumental":4, "International":5, "Pop": 6, "Rock": 7}

	song_dict = dict(zip(song_id, genre_list));

	rename_and_save(args.data_dir, args.output_dir, song_dict, genre_dict)


