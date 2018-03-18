"""
This script builds up the dataset for test.
It is a modification of the "build_dataset.py" provided in the project starter for pythorch

Note that we assume that all our data comes from the same distribution
As a result, we split the data into train/dev/test sets given a ratio. ie if ratio is 0.8, 
then 0.8 of the data will be used for training (train/dev (0.9/0.1)) and the rest for test

Future modifications
 -- allow for test to come from a different distribution ie. just split data into train and dev

"""

import os
import argparse
import random

from PIL import Image 
from tqdm import tqdm

SIZE = 128 #Using 128x128 pixels for now. with 50pix/s, this capture ~ 2.6s of song

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/sliced_specs', help="Directory with the raw spectrograms")
parser.add_argument('--train_fraction', default= 0.95, help="Percentage of dataset to be used for training")
parser.add_argument('--output_dir', default='data/128x128_specs', help="Where to write the new data")

def resize_and_save(fname, output_dir, size=SIZE):
	im = Image.open(fname)
	im = im.resize((size, size), Image.BILINEAR)
	im.save(os.path.join(output_dir, fname.split('/')[-1]))



if __name__ == '__main__':
	args = parser.parse_args()

	raw_specs = args.data_dir
	out_specs = args.output_dir
	train_fract = args.train_fraction

	assert os.path.isdir(raw_specs), "Couldn't find the dataset at {}".format(raw_specs)
	fnames = os.listdir(raw_specs)
	fnames = [os.path.join(raw_specs, f) for f in fnames if f.endswith(".png")]

	#Split the files into the different sets. Use seed to get consistent results on all runs
	random.seed(230)
	fnames.sort()
	random.shuffle(fnames)

	split = int(train_fract*len(fnames))
	train_dev_fnames = fnames[:split]
	test_fnames = fnames[split:]
	split = int(0.95*len(train_dev_fnames))
	train_fnames = train_dev_fnames[:split]
	dev_fnames = train_dev_fnames[split:]

	fnames = {'train': train_fnames, 'dev':dev_fnames, 'test':test_fnames}

	if not os.path.exists(out_specs):
		os.mkdir(out_specs)
	else:
		print("Warning: output dir {} already exists".format(out_specs))

	for split in ['train', 'dev', 'test']:
		output_dir = os.path.join(out_specs, '{}_specs'.format(split))

		if not os.path.exists(output_dir):
			os.mkdir(output_dir)
		else:
			print("Warning: output dir {} already exists".format(output_dir))

		print("Processing {} data, saving preprocessed data to {}".format(split, output_dir))

		for fname in tqdm(fnames[split]):
			resize_and_save(fname, output_dir, size=SIZE)

	print("Phew! I am finally done building the dataset!")






