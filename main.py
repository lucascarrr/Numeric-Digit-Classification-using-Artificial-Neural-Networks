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
from matplotlib import pyplot as plt
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

    #This code was wrapped in a for loop for testing purposes, if you want to measure the results of 
    # x-amount of models, change the range of the for loop to x
    for i in range (1):
        model_lists = [
        models.Model_1(),
        models.Model_2(),
        models.Model_3(),
        models.Model_4(),
        models.Model_5(),
        ]

        model = model_lists[model_number]
        if model_number == 3:                                                                        #-1 for index offput
            optimizer = torch.optim.Adam(model.parameters(),lr=model.learning_rate)
            optimizer_string = "Adam"
        else: 
            optimizer = torch.optim.SGD(model.parameters(), lr = model.learning_rate)
            optimizer_string = "SGD"
            
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
        iterations = 0
        for e in range (model.epochs):
            print (f"Epoch number: {e+1}\n")
            mt.train_model(model, training_data, optimizer, iterations)
            iterations += 1
            mt.validate_model(model, validation_data, iterations)
            print ("---"*11)

        # #plotting the loss graph
        # fig = plt.figure()
        # plt.plot(mt.training_examples_seen, mt.loss_values, color = 'blue')
        # plt.xlabel('Number of Training Examples Seen (10 000)')
        # plt.ylabel('Loss ')  
        # plt.savefig(('loss_'+str(model_number)+'.png'))

        # #plotting the accuracy graph
        # fig = plt.figure()
        # plt.plot(mt.accuracy_values_seen, mt.accuracy_values, color = 'green')
        # plt.xlabel('Number of Training Examples Seen')
        # plt.ylabel('Accuracy (%) ')  
        # plt.savefig(('accuracy_'+str(model_number)+'.png'))

        print("Finished Training!")
        x = mt.test_model(model, test_data)
        summation += x
    
    #print (summation, summation/x)        #uncomment this line, and change the value of x in order to get average performance (from the for loop above)

    response = ""
    while True:
        response = input(("Please enter a filepath/'exit to terminate:")+"\n")
        if response == 'exit': exit(0)
        
        # converting jpeg to tensor
        img = Image.open(response)
        convert_to_tensor = transforms.ToTensor()
        img = convert_to_tensor(img)

        # getting prediction from model
        model.eval()
        pred = model(img)
        pred = pred.detach().numpy()
        print("Classifier: ", np.argmax(pred))

