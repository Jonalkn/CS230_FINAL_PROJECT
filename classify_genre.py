"""
This script classifies a given song into it's genres,

It implements a voting mechanism in which the different slices from a single track
are classified independently. The popular of the genres of each slice is used.

More over, it should plot a bar graph showing the confidence levels for the different
genres.

For example if for a particular song, the genre #1 is found as the most likely with 60%
confidence the bar graph will be something like this:

Label :  Confidence
 0.0     *
 1.0     ************
 2.0     ***
 3.0     
 4.0     *
 5.0     **
 6.0 
 7.0     *

"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/128x128_specs', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/Model_5', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def get_genre_confidence(model, dataloader, params):
    """Get a vector with the probabilities per genre for a given song

    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        params: (Params) hyperparameters

    Returns:
    genre_confs : a numpy array of num_classes x 1 of the confidences of the model regarding
                  the song genre
    """

    # set model to evaluation mode
    model.eval()

    num_classes = params.num_classes

    # A matrix with confidences for every spectrogram slice
    confidence_matrix = np.empty((num_classes, 0)) 


    # determine the confidences over the batches
    for data_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch = data_batch.cuda(async=True)
        # fetch the next evaluation batch
        data_batch = Variable(data_batch)
        
        # compute model output
        output_batch = model(data_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()

        #Append the outputs batch to the confidence matrix 
        confidence_matrix = np.append(confidence_matrix, output_batch.T, axis = 1)

    # compute mean of all metrics in summary
    genre_confs = np.exp(np.mean(confidence_matrix, axis = 1))

    return genre_confs/np.sum(genre_confs)


if __name__ == '__main__':
    """
        Classify a song into a genre given different slices of the song.

        About the naming convention
        In this implementation, we use tmp (temporary) since this data is temporary for just a particular song that we 
        are trying to classify
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'classify.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['tmp'], args.data_dir, params)
    tmp_dl = dataloaders['tmp']

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    logging.info("Starting classification")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    genre_confidences = get_genre_confidence(model, tmp_dl, params)
    
    max_idx = np.argmax(genre_confidences)
    if genre_confidences[max_idx] >= params.confidence_threshold:
        print("This track is definitely " + utils.label_to_genre[max_idx])
        print("I am "+ str(genre_confidences[max_idx]*100) + "% sure.")
    else: 
        print("This is a hard one! Take a look at the bars")
    

    #Plot the bar graph here!
    utils.plot_bar_graph(genre_confidences)
