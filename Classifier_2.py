# CSC3022 Assignment 4: ANN image classification 
# Classifier 1 - no optimization. Worst performance expected
# Lucas Carr    ||  CRRLUC003
# 25 April 2022

# Libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

# hyper parameters
network_layers = [784, 300, 10]
learning_rate = 1e-1
epochs = 5
batch_size = 32

# getting MNIST data
training_data = datasets.MNIST(
    'data', 
    train=True, 
    download=False, 
    transform=ToTensor(),
    )

training_dataloader = DataLoader(
    training_data, 
    batch_size = batch_size,
)

test_data = datasets.MNIST(
    'data', 
    train=False,
    download=False,
    transform=ToTensor(),
)

test_dataloader = DataLoader(
    test_data,
    batch_size = batch_size,
)

#training
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X, y

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
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

# model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(network_layers[0], network_layers[1]),
            nn.ReLU(),
            nn.Linear(network_layers[1], network_layers[1]),
            nn.ReLU(),
            # nn.Linear(network_layers[2], network_layers[2]),
            # nn.ReLU(),
            # nn.Linear(network_layers[3], network_layers[3]),
            # nn.ReLU(),
            nn.Linear(network_layers[1], network_layers[2]),

        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
print (model)

# extra
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for e in range(epochs):
    print(f"Epoch {e+1}\n-------------------------------")
    train(training_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Finished Learning!")
print("Please enter a filepath to test: ")