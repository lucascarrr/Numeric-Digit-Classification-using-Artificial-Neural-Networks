""" CSC3022 Assignment 4: ANNs
    CRRLUC003 || Lucas Carr

    Classifier_1: 
        Loss Function: MSE Loss
        Activation Function: ReLU
        Optimizer: Adam
        Batch Size: 32
        Learning Rate: 1e-1 (0.00001)
"""

import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

# hyper parameters
network_layers = [784, 300, 10]
learning_rate = 1e-1
epochs = 5
batch_size = 10


# getting MNIST data
training_data = datasets.MNIST('data', train=True, download=False, transform=ToTensor())
training_dataloader = DataLoader(training_data, batch_size = batch_size)

test_data = datasets.MNIST('data', train=False, download=False, transform=ToTensor())
test_dataloader = DataLoader(test_data, batch_size = batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(network_layers[0], network_layers[1]),
            nn.ReLU(),
            nn.Linear(network_layers[1], network_layers[1]),
            nn.ReLU(),
            nn.Linear(network_layers[1], network_layers[2]),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, network, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X,y
        prediction = network(X)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, network):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X, y
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    network = NeuralNetwork()
    print (network)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    for e in range(epochs):
        print(f"Epoch number {epochs+1}\n____________________________")
        train(training_dataloader, network, loss_fn, optimizer)
        test(test_dataloader, network)

