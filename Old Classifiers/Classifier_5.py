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
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from PIL import Image

# hyper parameters
network_layers = [784, 100, 10]
learning_rate = 1e-3
epochs = 5
batch_size = 32

# getting MNIST data
training_data = datasets.MNIST('data', train=True, download=False, transform=ToTensor())            #training data (size = 60 000), split up into training, validation
test_data = datasets.MNIST('data', train=False, download=False, transform=ToTensor())               #test data (unseen by the model)
training_data, validation_data = data.random_split(training_data, (48000, 12000))                   #splitting the training data into training + validation

# loading datasets
training_dataloader = data.DataLoader(training_data, batch_size = batch_size, shuffle = True)
validation_dataloader = data.DataLoader(validation_data, batch_size = batch_size, shuffle = True)
test_dataloader = data.DataLoader(test_data, batch_size = batch_size,  shuffle = True)

# print (len(training_data), len(validation_data), len(test_data))                                  #checking lengths of data, 48 000 / 12 000 / 1000

def train_network(training_data, network, loss_function, optimizer):
    size = len(training_data.dataset)
    for training_batch, (X, y) in enumerate(training_data):
        pred = network(X)                       #prediction
        loss = loss_function(pred, y)                 #calc loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if training_batch % 100 == 0:
            loss, current = loss.item(), training_batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def validate_network(validation_data, network, loss_function):
        size = len(validation_data.dataset)
        network.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in validation_data:
                pred = network(X)
                test_loss += loss_function(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(f"Validation Accuracy Rate: {(100*(correct)):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def test_network(test_data, network):
    size = len(test_data.dataset)
    network.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_data:
            pred = network(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(f"Test Accuracy Rate: {(100*(correct)):>0.1f}")

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
        train_network(training_dataloader, network, loss_fn, optimizer)
        validate_network(validation_dataloader, network, loss_fn)
    
    test_network(test_dataloader, network)

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

        



        

    