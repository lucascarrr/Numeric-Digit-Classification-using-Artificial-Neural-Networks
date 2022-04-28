
from torch import nn
import torch.nn.functional as F
"""
Model 1:
    - 784, 32, 10
    - Learning Rate: 1e-1 (0.1)
    - Epochs: 20
    - Batch Size = 100
    - Activation Function: Sigmoid 
    - Loss Function: CrossEntropyLoss
    - Optimizer: SGD
"""
class Model_1(nn.Module):
    model_name = "Model 1"
    model_layers = [784, 32, 10]
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

"""
Model 2:
    - 784, 512, 10
    - Learning Rate: 1e-1 (0.1)
    - Epochs: 20
    - Batch Size = 64
    - Activation Function: ReLU 
    - Loss Function: CrossEntropyLoss
    - Optimizer: SGD
"""
class Model_2(nn.Module):
    model_name = "Model 2"
    model_layers = [784, 512, 10]
    learning_rate = 1e-1
    epochs = 20
    batch_size = 64

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

"""
Model 3:
    - 784, 512, 10
    - Learning Rate: 1e-1 (0.1)
    - Epochs: 20
    - Batch Size = 32
    - Activation Function: ReLU 
    - Loss Function: CrossEntropyLoss
    - Optimizer: SGD
"""
class Model_3(nn.Module):
    model_name = "Model 3"
    model_layers = [784, 512, 10]
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

"""
Model 4:
    - 784, 512, 10
    - Learning Rate: 1e-1 (0.1)
    - Epochs: 5
    - Batch Size = 32
    - Activation Function: ReLU 
    - Loss Function: CrossEntropyLoss
    - Optimizer: SGD
"""
class Model_4(nn.Module):
    model_name = "Model 4"
    model_layers = [784, 512, 10]
    learning_rate = 1e-1
    epochs = 5
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

"""
Model 5:
    - 784, 100, 10
    - Learning Rate: 1e-1 (0.1)
    - Epochs: 5
    - Batch Size = 32
    - Activation Function: ReLU 
    - Loss Function: CrossEntropyLoss
    - Optimizer: SGD

    Added batch noramlizing
"""
class Model_5(nn.Module):
    model_name = "Model 5"
    model_layers = [784, 100, 10]
    learning_rate = 1e-1
    epochs = 5
    batch_size = 32

    loss_function = nn.CrossEntropyLoss()

    def __init__(self):
        super(Model_5, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(Model_5.model_layers[0], Model_5.model_layers[1]),
            nn.ReLU(),
            nn.BatchNorm1d(Model_5.model_layers[1]),
            nn.Linear(Model_5.model_layers[1], Model_5.model_layers[1]),
            nn.ReLU(),
            nn.BatchNorm1d(Model_5.model_layers[1]),
            nn.Linear(Model_5.model_layers[1], Model_5.model_layers[2]),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Model_6(nn.Module):
    model_name = "Model 6"
    model_layers = [784, 100, 10]
    learning_rate = 1e-3
    epochs = 5
    batch_size = 32
    loss_function = nn.CrossEntropyLoss()
    
    def __init__(self):
        super(Model_6, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)