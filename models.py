import numpy as np 
import torch
from torch import nn
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from PIL import Image

class Model_1(nn.Module):
    model_name = "Model 1"
    model_layers = [784, 500, 10]
    learning_rate = 1e-1
    epochs = 20
    batch_size = 100

    loss_function = nn.CrossEntropyLoss()

    def __init__(self):
        super(Model_1, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(Model_1.model_layers[0], Model_1.model_layers[1]),
            nn.Sigmoid(),
            nn.Linear(Model_1.model_layers[1], Model_1.model_layers[1]),
            nn.Sigmoid(),
            nn.Linear(Model_1.model_layers[1], Model_1.model_layers[2]),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Model_2(nn.Module):
    model_name = "Model 2"
    model_layers = [784, 500, 10]
    learning_rate = 1e-1
    epochs = 20
    batch_size = 100

    loss_function = nn.CrossEntropyLoss()

    def __init__(self):
        super(Model_2, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(Model_2.model_layers[0], Model_2.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_2.model_layers[1], Model_2.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_2.model_layers[1], Model_2.model_layers[2]),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Model_3(nn.Module):
    model_name = "Model 3"
    model_layers = [784, 100, 10]
    learning_rate = 1e-1
    epochs = 20
    batch_size = 32

    loss_function = nn.CrossEntropyLoss()

    def __init__(self):
        super(Model_3, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(Model_3.model_layers[0], Model_3.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_3.model_layers[1], Model_3.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_3.model_layers[1], Model_3.model_layers[2]),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Model_4(nn.Module):
    model_name = "Model 4"
    model_layers = [784, 100, 10]
    learning_rate = 1e-1
    epochs = 25
    batch_size = 32

    loss_function = nn.CrossEntropyLoss()

    def __init__(self):
        super(Model_4, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(Model_4.model_layers[0], Model_4.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_4.model_layers[1], Model_4.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_4.model_layers[1], Model_4.model_layers[2]),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Model_5(nn.Module):
    model_name = "Model 5"
    model_layers = [784, 100, 10]
    learning_rate = 1e-1
    epochs = 25
    batch_size = 32

    loss_function = nn.CrossEntropyLoss()

    def __init__(self):
        super(Model_5, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(Model_5.model_layers[0], Model_5.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_5.model_layers[1], Model_5.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_5.model_layers[1], Model_5.model_layers[2]),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits