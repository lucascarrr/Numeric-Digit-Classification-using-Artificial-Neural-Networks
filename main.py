# CSC3022 Assignment 4
# Lucas Carr
# CRRLUC003
# main.py

import numpy as np 
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from PIL import Image

import sys
import models
from model_trainer import model_trainer as mt


training_data = datasets.MNIST('data', train=True, download=True, transform=ToTensor())            #training data (size = 60 000), split up into training, validation
test_data = datasets.MNIST('data', train=False, download=True, transform=ToTensor())               #test data (unseen by the model)
training_data, validation_data = data.random_split(training_data, (48000, 12000))  


if __name__=='__main__':
    model_number = sys.argv[1]
    model_number = int(model_number) - 1

    summation = 0
    #testing
    for i in range (1):
        model_lists = [
        models.Model_1(),
        models.Model_2(),
        models.Model_3(),
        models.Model_4(),
        models.Model_5(),
        ]

        model = model_lists[model_number]
        if model_number < 3:        #-1 for index offput
            optimizer = torch.optim.SGD(model.parameters(), lr = model.learning_rate)
            optimizer_string = "SGD"
        else: 
            optimizer = torch.optim.Adam(model.parameters(),lr=model.learning_rate,betas=(0.9,0.999),
                eps=1e-08,weight_decay=0,amsgrad=False)
            optimizer_string = "Adam"
            
        #Printing Model Details
        print (model.model_name,":", sep="")
        print ("Topology: ", model.model_layers)
        print ("Activation Function: ", model.activation_function)
        print ("Loss Function: ", model.loss_function)
        print ("Optimizer: ", optimizer_string)
        print ("Batch Size: ", model.batch_size)
        print ("Learning Rate: ", model.learning_rate)
        print ("---"*11)

        #Training
        for e in range (model.epochs):
            print (f"Epoch number: {e+1}\n")
            mt.train_model(model, training_data, optimizer)
            mt.validate_model(model, validation_data)
            print ("---"*11)

        print("Finished Training")
        x = mt.test_model(model, test_data)
        summation += x
    
    print (summation, summation/20)

    response = ""
    while True:
        response = input(("Please enter a filepath/'exit to terminate:")+"\n")
        if response == 'exit': exit(0)
        
        img = Image.open(response)
        convert_to_tensor = transforms.ToTensor()
        img = convert_to_tensor(img)

        model.eval()
        pred = model(img)
        pred = pred.detach().numpy()
        print("Classifier: ", np.argmax(pred))

