""" CSC3022 Assignment 4: ANNs
    CRRLUC003 || Lucas Carr

    Classifier_1: 
        Loss Function: Cross Entropy Loss
        Activation Function: Sigmoid
        Optimizer: SGD
        Batch Size: 100
        Learning Rate: 1e-1 (0.00001)
"""

import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose
from PIL import Image

# hyper parameters
network_layers = [784, 600, 10]
learning_rate = 1e-1
epochs = 1
batch_size = 100

# getting MNIST data
training_data = datasets.MNIST('data', train=True, download=False, transform=ToTensor())
test_data = datasets.MNIST('data', train=False, download=False, transform=ToTensor())

training_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = batch_size,  shuffle = True)

# train network
def train(dataloader, network, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = network(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# test network
def test(dataloader, network):
    size = len(dataloader.dataset)
    network.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = network(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy Rate: {(100*(correct)):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Define network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(network_layers[0], network_layers[1]),
            nn.Sigmoid(),
            nn.Linear(network_layers[1], network_layers[1]),
            nn.Sigmoid(),
            nn.Linear(network_layers[1], network_layers[2]),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__=='__main__':
    network = NeuralNetwork()
    print(network)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_dataloader, network, loss_fn, optimizer)
        test(test_dataloader, network)

    response = ""
    while response != "exit":
        response = input(("Please enter a filepath/'exit to terminate:")+"\n")
        img = Image.open(response)
        convert_to_tensor = transforms.ToTensor()
        img = convert_to_tensor(img)
        pred = network(img)
        print(pred)
        


        

    