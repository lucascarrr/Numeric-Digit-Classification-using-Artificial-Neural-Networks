

## CSC3022 Assignment 4

### Numeric Digit Classification using Artificial Neural Networks

### Lucas Carr || CRRLUC003

#### Files/Program Execution

> **Makefile**
>
> Creates the virtual environment for this program. Prior to any other execution, make sure you run the make file, and enter the virtual environment by inputting the following into your terminal:
>
> ```shell
> make	#venv created
> 
> source ./venv/bin/activate	#this will activate the virtual environment
> ```

>**Requirements.txt**
>
>File contains the Python libraries for the virtual environment. 

> **models.py** 
>
> This file contains seperate classes for each neural network model. They are referenced by the notation ‘Model_<model number>’. Each model class contains all the parameters/functions/topology of the model - except the optimiser, which is decided in main.py. 

> **model_trainer.py** 
>
> This file contains the training, validating, and testing methods which each model will utilize. Methods are: *train_model*( ), *validate_model*( ), and *test_model*( ), respectively. 

> **main.py**
>
> File contains the main method. Each model class is imported and stored in a list, when running this file you can choose which model you would like to run. 
>
> ```shell
> python main.py 4 						#this will run the program with model 4
> ```
>
> Once training and validation are complete, the program will prompt you to enter in an individual MNIST .jpeg file (the path), and return the model’s classification of your file. 
>
> ```shell
> 'Please enter a filepath/'exit' to terminate:'
> 	> some_folder/some_image.jpeg
> 'Classification: 3'
> ```
>
> 

