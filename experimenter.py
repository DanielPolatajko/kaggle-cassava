# Utility class to build experiments, allowing easy changing of hyperparams, models, etc

import torch
import time
import copy

from tqdm import tqdm

class Experimenter:
    """
    Class to construct Pytorch experiment
    """

    def __init__(self, model, device, loss_function, optimiser, lr_scheduler, train_data, val_data, num_epochs):
        """
        Add some fun hyperparameters
        :param model: The model we want to experiment with
        :param device: The device we want to run the experiment on
        :param loss_function: The loss function to optimise
        :param optimiser: The optimiser to use
        :param lr_scheduler: Learning rate scheduler
        :param train_data: Training data for the learning task
        :param val_data: Validation data for the learning task
        """
        self.model = model
        self.device = device
        self.loss_function = loss_function
        self.optimiser = optimiser
        self.lr_scheduler = lr_scheduler
        self.train_data = train_data
        self.val_data = val_data
        self.num_epochs = num_epochs

    def training_epoch(self):
        """
        Method to execute a training epoch
        :return:
        """

        # initiliase epoch loss and predictions for accuracy
        cumulative_loss = 0.0
        correct_preds = 0

        with tqdm(self.train_data, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimiser.zero_grad()

                # forward pass
                with torch.set_grad_enabled(True):
                    # Get model outputs and calculate loss
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)

                    # make predictions
                    _, preds = torch.max(outputs, 1)

                    # backward pass and parameter optimisation
                    loss.backward()
                    self.optimiser.step()

                # update loss and predictions
                cumulative_loss += loss.item() * inputs.size(0)
                correct_preds += torch.sum(preds == labels.data).item()

                tepoch.set_postfix(loss=loss.item())

            epoch_loss = cumulative_loss / len(self.train_data.dataset)
            epoch_acc = correct_preds / len(self.train_data.dataset)

        return epoch_loss, epoch_acc

    def validation_epoch(self):
        """
        Method to execute a validation epoch
        :return:
        """

        # initiliase epoch loss and predictions for accuracy
        cumulative_loss = 0.0
        correct_preds = 0

        for inputs, labels in self.train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            self.optimiser.zero_grad()

            # forward pass
            with torch.set_grad_enabled(False):
                # Get model outputs and calculate loss
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)

                # make predictions
                _, preds = torch.max(outputs, 1)

            # update loss and predictions
            cumulative_loss += loss.item() * inputs.size(0)
            correct_preds += torch.sum(preds == labels.data)

        epoch_loss = cumulative_loss / len(self.train_data.dataset)
        epoch_acc = correct_preds / len(self.train_data.dataset)

        return epoch_loss, epoch_acc


    def train(self):
        """
        Method to train the model based on the given hyperparameters
        :return:
        """
        start = time.time()

        # keep losses and accuracies to plot training afterwards
        stats = {
            'train_losses': [],
            'train_accs': [],
            'val_losses': [],
            'val_accs': []
        }

        best_model_params = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}\n")

            # set model to training mode
            self.model.train()

            # do a training epoch to get loss and acc, and record them
            train_loss, train_acc = self.training_epoch()
            stats['train_losses'].append(train_loss)
            stats['train_accs'].append(train_acc)

            print(f"Training loss - Epoch {epoch+1}/{self.num_epochs}: {train_loss}")
            print(f"Training accuracy - Epoch {epoch+1}/{self.num_epochs}: {train_acc}")

            # set model to training mode
            self.model.eval()

            # do a training epoch to get loss and acc, and record them
            val_loss, val_acc = self.validation_epoch()
            stats['val_losses'].append(train_loss)
            stats['val_accs'].append(train_acc)

            print(f"Validation loss - Epoch {epoch+1}/{self.num_epochs}: {val_loss}")
            print(f"Validation accuracy - Epoch {epoch+1}/{self.num_epochs}: {val_acc}")

            # keep a copy of the best performing model parameters
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_params = copy.deepcopy(self.model.state_dict())

            print()

        end = time.time() - start
        print(f"Training finished in {end:.0f}m {end:.0f}s")
        print(f"Best validation accuracy: {best_acc:4f}")

        # load best model weights
        self.model.load_state_dict(best_model_params)
        return self.model, stats