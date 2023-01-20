# -*- coding: utf-8 -*-
import cProfile
import click
import logging
import pstats

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import torch
from torch import nn, optim

from model import MyAwesomeModel


@click.command()
@click.argument('model_dir', type=click.Path())

def main(model_dir):
    print("Training...")

    model = MyAwesomeModel()

    # Load train set
    data_train = np.load("data/processed/train.npz")

    # Create Dataset from the npz files
    class NpzDataset(torch.utils.data.Dataset):
        # Class to create a torch Dataset class for inputting into Dataloader
        def __init__(self, data):
            self.img = data["images"]
            self.lab = data["labels"]

        def __getitem__(self, idx):
            return (self.img[idx], self.lab[idx])

        def __len__(self):
            return len(self.img)

    dataset_train = NpzDataset(data_train)

    # Create dataloader from the Dataset
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    # Training loop
    epochs = 30
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1).float()

            # TODO: Training pass
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            print(f"Training loss: {running_loss/len(trainloader)}")

            #checkpoint = {'input_size': 784, 'output_size': 10, 'hidden_layers': [each.out_features for each in model.hidden_layers],'state_dict': model.state_dict()}
            #torch.save(checkpoint, r"models/checkpoint.pth")
            torch.save(model.state_dict(), r"models/checkpoint.pth")
    
    print("Finished training.")
    print("checkpoint.pth saved in ./models folder")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # Profiling with cProfile
    cProfile.run("main()", "profiling/profile.dat") # creates profile.dat

    with open("profiling/profile_time.txt", "w") as f: # creates profile_time.txt
        p = pstats.Stats("profiling/profile.dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("profiling/profile_calls.txt", "w") as f: # creates profile_calls.txt
        p = pstats.Stats("profiling/profile.dat", stream=f)
        p.sort_stats("calls").print_stats()
    print("Profiling executed and profiling info saved in ./profiling folder")

    main()