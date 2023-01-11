# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
#input_filepath = 'data/raw'
#output_filepath = 'data/processed'

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    images, labels = concat_npz(input_filepath)
    
    # Save the concatenated lists to a new .npz file
    np.savez_compressed(output_filepath + "/train.npz", images=np.array(images), labels=np.array(labels))

def concat_npz(input_path):
    # Function to concatenate 5 training npz files into one
    images = []
    labels = []
    for file in range(0, 5):
        with np.load(input_path + "/train_" + str(file) + ".npz") as data1:
            img1 = data1["images"]
            for img11 in img1:
                images.append(img11)
            lab1 = data1["labels"]
            for lab11 in lab1:
                labels.append(lab11)
    return images, labels
        

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
