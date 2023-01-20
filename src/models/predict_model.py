import numpy as np
import torch
from model import MyAwesomeModel

model = MyAwesomeModel()

# Load the state dict
state_dict = torch.load(r"models/checkpoint.pth")
#print(state_dict)

# Load state dict into the network
model.load_state_dict(state_dict)
print(model)


# Load test set
data_test = np.load("data/raw/test.npz")

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

dataset_test = NpzDataset(data_test)

# Create dataloader from the Dataset
testloader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True)


# Test out your network!
dataiter = iter(testloader)
images, labels = next(dataiter)
img = images[0]
# Convert 2D image to 1D vector
img = img.view(1, 784)


# Calculate the class probabilities (softmax) for img
with torch.no_grad():
    output = model.forward(img.float())
    print('output is', output)

# I don't know why we do this. I guess it is for the following helper function
ps = torch.exp(output)
print('ps is:', ps) 