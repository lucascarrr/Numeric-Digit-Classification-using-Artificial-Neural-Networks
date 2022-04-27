""" CSC3022 Assignment 4: ANNs
    CRRLUC003 || Lucas Carr

    Classifier_5: 
        Loss Function: Cross-Entropy Loss
        Activation Function: ReLU
        Optimiser: SGD
        Topology: [784, 100, 10]
        Learning Rate: 1e-3 (0.1)
        Weight Initialisation: Random
        Batch Size: 32
        Epochs: 5
    
    Changes:
        - Learning Rate (1e-1 -> 1e-3)
        
    Problems:
        - Good Accuracy
        - Too quick to converge
        - Overfitting Data
    
"""
import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose
from PIL import Image

# hyper parameters
network_layers = [784, 100, 10]
learning_rate = 1e-3
epochs = 5
batch_size = 32

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

        # l2_lambda = 0.001
        # l2_norm = sum(p.pow(2.0).sum()
        #     for p in network.parameters())
        # loss = loss + l2_lambda * l2_norm

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    network.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = network(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Train Accuracy Rate: {(100*(correct)):>0.1f}%, Avg loss: {test_loss:>8f} \n")

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
    print(f"Test Accuracy Rate: {(100*(correct)):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

# Define network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(network_layers[0], network_layers[1]),
            nn.ReLU(),
            nn.BatchNorm1d(network_layers[1]),
            nn.Linear(network_layers[1], network_layers[1]),
            nn.ReLU(),
            nn.BatchNorm1d(network_layers[1]),
            nn.Linear(network_layers[1], network_layers[2]),
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
    while True:
        response = input(("Please enter a filepath/'exit to terminate:")+"\n")
        if response == 'exit': exit(0)
        
        img = Image.open(response)
        convert_to_tensor = transforms.ToTensor()
        img = convert_to_tensor(img)

        network.eval()
        pred = network(img)
        pred = pred.detach().numpy()
        print("Classifier: ", np.argmax(pred))

        



        

    