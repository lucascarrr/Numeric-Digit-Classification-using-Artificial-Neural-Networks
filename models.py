# CSC3022 Assignment 4
# Lucas Carr
# CRRLUC003
# models.py

# Inspiration taken from:     https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch.nn.functional as F
from torch import dropout, nn

"""
Model 1:
    - 784, 32, 10
    - Activation Function: Sigmoid 
    - Loss Function: CrossEntropyLoss
    - Optimizer: SGD
    - Learning Rate: 0.01
    - Epochs: 10
    - Batch Size = 150


"""
class Model_1(nn.Module):
    model_name = "Model 1"
    model_layers = [784, 32, 10]
    learning_rate = 1e-2
    epochs = 10
    batch_size = 150
    activation_function = "Sigmoid"
    loss_function = nn.CrossEntropyLoss()
    
    def __init__(self):
        super(Model_1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(Model_1.model_layers[0], Model_1.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_1.model_layers[1], Model_1.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_1.model_layers[1], Model_1.model_layers[2]),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    model_name = "Model 1"
    model_layers = [784, 32, 10]
    learning_rate = 1e-2
    epochs = 10
    batch_size = 150
    activation_function = "Sigmoid"
    loss_function = nn.CrossEntropyLoss()
    
    def __init__(self):
        super(Model_1, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(Model_1.model_layers[0], Model_1.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_1.model_layers[1], Model_1.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_1.model_layers[1], Model_1.model_layers[2]),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits
"""
Model 2:
    - 784, 32, 10
    - Learning Rate: 1e-1 (0.1)     (change)
    - Epochs: 16
    - Batch Size = 150 
    - Activation Function: ReLU     (change)
    - Loss Function: CrossEntropyLoss
    - Optimizer: SGD
"""
class Model_2(nn.Module):
    model_name = "Model 2"
    model_layers = [784, 32, 10]
    learning_rate = 1e-1
    epochs = 16
    batch_size = 150
    activation_function = "ReLU"
    loss_function = nn.CrossEntropyLoss()

    def __init__(self):
        super(Model_2, self).__init__()

        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(Model_2.model_layers[0], Model_2.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_2.model_layers[1], Model_2.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_2.model_layers[1], Model_2.model_layers[2]),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits

"""
Model 3:
    - [784, 512, 256, 10]           (change)
    - Learning Rate: 1e-1 (0.1)     (change)
    - Epochs: 8
    - Batch Size = 32               (change)
    - Activation Function: ReLU 
    - Loss Function: NLLL           (change)
    - Optimizer: SGD               
"""
class Model_3(nn.Module):
    model_name = "Model 3"
    model_layers = [784, 512, 256, 10]
    learning_rate = 0.1
    epochs = 8
    batch_size = 32
    activation_function = "ReLU"
    loss_function = nn.NLLLoss()
    def __init__(self):
        super(Model_3, self).__init__()

        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(Model_3.model_layers[0], Model_3.model_layers[1]),
            nn.BatchNorm1d(Model_3.model_layers[1]),
            nn.ReLU(),

            nn.Linear(Model_3.model_layers[1], Model_3.model_layers[2]),
            nn.BatchNorm1d(Model_3.model_layers[2]),
            nn.ReLU(),

            nn.Linear(Model_3.model_layers[2], Model_3.model_layers[2]),
            nn.BatchNorm1d(Model_3.model_layers[2]),
            nn.ReLU(),

            nn.Linear(Model_3.model_layers[2], Model_3.model_layers[3]),
            nn.LogSoftmax(dim=1),

        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits

"""
Model 4:
    - 784, 400, 10                          (change)
    - Learning Rate: 0.01                   (change)
    - Epochs: 10                            (change)
    - Batch Size = 32
    - Activation Function: ReLU, 
        ReLU, ReLU, LogSoftMax              (change)
    - Loss Function: NLLLoss                (change)
    - Optimizer: Adam                       (change)
"""
class Model_4(nn.Module):
    model_name = "Model 4"
    model_layers = [784, 400, 10]
    learning_rate = 0.01
    epochs = 10
    batch_size = 32
    activation_function = "ReLU"
    loss_function = nn.NLLLoss()

    def __init__(self):
        super(Model_4, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(Model_4.model_layers[0], Model_4.model_layers[1]),
            nn.BatchNorm1d(Model_4.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_4.model_layers[1], Model_4.model_layers[1]),
            nn.BatchNorm1d(Model_4.model_layers[1]),
            nn.ReLU(),
            nn.Linear(Model_4.model_layers[1], Model_4.model_layers[2]),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits

"""
Model 5:
    - 784, 100, 10
    - Learning Rate: 1e-1 (0.1)
    - Epochs: 5
    - Batch Size = 32
    - Activation Function: ReLU 
    - Loss Function: CrossEntropyLoss
    - Optimizer: SGD

    Inspiration taken from: https://www.kaggle.com/code/sdelecourt/cnn-with-pytorch-for-mnist/notebook
"""
class Model_5(nn.Module):
    model_name = "Model 5"
    learning_rate = 0.02
    model_layers = [1, 32, 64, 256, 576, 10]
    epochs = 10
    batch_size = 32
    activation_function = "ReLU"
    loss_function = nn.CrossEntropyLoss()

    def __init__(self):
        super(Model_5, self).__init__()
        self.conv1 = nn.Conv2d(Model_4.model_layers[0], Model_4.model_layers[1], kernel_size=5)
        self.conv2 = nn.Conv2d(Model_4.model_layers[1], Model_4.model_layers[1], kernel_size=5)
        self.conv3 = nn.Conv2d(Model_4.model_layers[1],Model_4.model_layers[2], kernel_size=5)

        self.fc1 = nn.Linear(Model_4.model_layers[4], Model_4.model_layers[3])
        self.fc2 = nn.Linear(Model_4.model_layers[3], Model_4.model_layers[5])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)

        x = x.view(-1,Model_4.model_layers[4] )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)