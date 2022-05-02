import torch
import torch.utils.data as data

# inspiration taken from:     https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

class model_trainer:

    #Used for plotting graphs
    training_examples_seen = []
    # loss_values = []
    # accuracy_values = []
    # accuracy_values_seen = []

    def train_model(model, training_data, optimizer, iteration):
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
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                # for creating graphs:
                # model_trainer.training_examples_seen.append((size*iteration) + current)
                # model_trainer.loss_values.append(loss)

    def validate_model(model, validation_data, iteration):
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
        # for creating graphs
        # model_trainer.accuracy_values_seen.append(48000*iteration)
        # model_trainer.accuracy_values.append(100*correct)

    def test_model(model, test_data):
        test_dataloader = data.DataLoader(test_data, batch_size = model.batch_size,  shuffle = True)
        model.eval()
        size = len(test_dataloader.dataset)
        correct = 0

        with torch.no_grad():
            for X, y in test_data:
                pred = model(X)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        correct /= size
        print(f"Test Accuracy Rate: {(100*(correct)):>0.1f}")
        return (100*correct)



