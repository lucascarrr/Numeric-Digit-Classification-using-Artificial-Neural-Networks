import numpy as np 
import torch
from torch import nn
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from PIL import Image

import sys
import models
from model_trainer import model_trainer as mt

training_data = datasets.MNIST('data', train=True, download=False, transform=ToTensor())            #training data (size = 60 000), split up into training, validation
test_data = datasets.MNIST('data', train=False, download=False, transform=ToTensor())               #test data (unseen by the model)
training_data, validation_data = data.random_split(training_data, (48000, 12000))  

if __name__=='__main__':
    model_number = sys.argv[1]
    model_number = int(model_number) - 1

    model_lists = [
        models.Model_1(),
        models.Model_2(),
        models.Model_3(),
        models.Model_4(),
        models.Model_5(),
        ]

    model = model_lists[model_number]
    optimizer = torch.optim.SGD(model.parameters(), lr = model.learning_rate)

    print (model.model_name)
    for e in range (model.epochs):
        mt.train_model(model, training_data, optimizer)
        mt.validate_model(model, validation_data)



