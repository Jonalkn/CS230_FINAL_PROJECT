import os
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/128x128_specs_5.12s', help="Directory containing the dataset")

if __name__ == '__main__':

	args = parser.parse_args()

	file_dir = args.data_dir

	if not os.path.exists(file_dir):
		print("Warning: source dir {} does not exists".format(file_dir))
		sys.exit("I quit!")

	[os.rename(os.path.join(file_dir, f), os.path.join(file_dir, f.split(".pn")[0] + "512s.png")) for f in os.listdir(file_dir) if f.endswith(".png")]



		
