import numpy as np 
import torch
from torch import nn
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from PIL import Image

class model_trainer:
    def train_model(model, training_data, optimizer):
        training_dataloader = data.DataLoader(training_data, batch_size = model.batch_size, shuffle = True)
        model.train()
        size = len(training_dataloader.dataset)
        
        for batch, (X, y) in enumerate(training_dataloader):
            prediction = model(X)
            loss = model.loss_function(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print("Loss:", loss, "current:", size)
    
    def validate_model(model, validation_data):
        validation_dataloader = data.DataLoader(validation_data, batch_size = model.batch_size, shuffle = True)
        model.eval()
        size = len(validation_dataloader.dataset)
        test_loss, correct = 0, 0
        
        with torch.no_grad():
            for X, y in validation_dataloader:
                pred = model(X)
                test_loss += model.loss_function(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        print(f"Validation Accuracy Rate: {(100*(correct)):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def test_model(model, test_data):
        test_dataloader = data.DataLoader(test_data, batch_size = model.batch_size,  shuffle = True)
    
        model.eval()
        size = len(test_dataloader.dataset)
        model.eval()

        with torch.no_grad():
            for X, y in test_dataloader:
                pred = model(X)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        correct /= size
        print(f"Test Accuracy Rate: {(100*(correct)):>0.1f}")



