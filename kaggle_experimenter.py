# to be run in kaggle kernels, runs on a single split of data as kaggle has time limit for training

# %% [code] {"_kg_hide-input":false}
# %% [code] {"_kg_hide-input":false}
# %% [code] {"_kg_hide-input":false}
# %% [code] {"_kg_hide-input":false}
# %% [code] {"_kg_hide-input":false}
# Utility class to build experiments, allowing easy changing of hyperparams, models, etc

import torch
import time
import copy

from tqdm import tqdm

from btloss import BiTemperedLogisticLoss
import torch.optim as optim
from cassava_data_utils import reshape_model


class Experimenter:
    """
    Class to construct Pytorch experiment
    """

    def __init__(self, model,  experiment_name, device, loss_function, optimiser, lr_scheduler, dataset, num_epochs, num_classes, split, early_stopping=True,
                 patience=3, tta=5):
        """
        Add some fun hyperparameters
        :param model: The model we want to experiment with
        :param model_name: The name for the model (to save)
        :param device: The device we want to run the experiment on
        :param loss_function: The loss function to optimise
        :param optimiser: The optimiser to use
        :param lr_scheduler: Learning rate scheduler
        :param dataset: A DatasetConstructor object that can be used to perform CV
        :param num_epochs: The number of training epochs
        :param early_stopping: Whether or not to stop training early if overfitting
        :param patience: How many epochs to tolerate poorer validation accuracy for
        """
        self.model = model
        self.experiment_name = experiment_name
        self.device = device
        self.loss_function = loss_function
        self.optimiser = optimiser
        self.lr_scheduler = lr_scheduler

        # which cv split to run on
        self.split = split

        self.dataset = dataset
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.one_hot = False
        self.tta = tta
        self.num_classes = num_classes
        self.one_hot = True

    def training_epoch(self, train_data):
        """
        Method to execute a training epoch
        :return:
        """

        # initiliase epoch loss and predictions for accuracy
        cumulative_loss = 0.0
        correct_preds = 0

        with tqdm(train_data, unit="batch") as tepoch:
            for inputs, labels in tepoch:

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimiser.zero_grad()

                # forward pass
                with torch.set_grad_enabled(True):
                    # Get model outputs and calculate loss
                    outputs = self.model(inputs)
                    if self.one_hot:
                        one_hot_labels = torch.zeros(len(labels), self.num_classes).cuda()
                        one_hot_labels[range(one_hot_labels.shape[0]), labels] = 1
                        loss = self.loss_function(outputs, one_hot_labels)
                    else:
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

            epoch_loss = cumulative_loss / len(train_data.dataset)
            epoch_acc = correct_preds / len(train_data.dataset)

        return epoch_loss, epoch_acc

    def validation_epoch(self, val_data):
        """
        Method to execute a validation epoch
        :return:
        """

        # initiliase epoch loss and predictions for accuracy
        cumulative_loss = 0.0
        correct_preds = 0

        with tqdm(val_data, unit="batch") as tepoch:

            for inputs, labels in tepoch:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimiser.zero_grad()

                tta_individuals = torch.zeros((len(labels), self.num_classes)).cuda()

                # we do test time augmentation for validation
                for tta_ix in range(self.tta):
                    # forward pass
                    with torch.set_grad_enabled(False):
                        # Get model outputs and calculate loss
                        outputs = self.model(inputs)
                        tta_individuals += outputs

                # average the model predictions against augmentations
                outputs = tta_individuals / self.tta

                # one hot encode the labels for bi-tempered logistic loss
                if self.one_hot:
                    one_hot_labels = torch.zeros(len(labels), self.num_classes).cuda()
                    one_hot_labels[range(one_hot_labels.shape[0]), labels] = 1
                    loss = self.loss_function(outputs, one_hot_labels)
                else:
                    loss = self.loss_function(outputs, labels)

                # make predictions
                _, preds = torch.max(outputs, 1)

                # update loss and predictions
                cumulative_loss += loss.item() * inputs.size(0)
                correct_preds += torch.sum(preds == labels.data)

                tepoch.set_postfix(loss=loss.item())

            epoch_loss = cumulative_loss / len(val_data.dataset)
            epoch_acc = correct_preds / len(val_data.dataset)

        return epoch_loss, epoch_acc

    def train(self):
        """
        Method to train the model based on the given hyperparameters
        :return:
        """


        stopped_early = False
        stopped_epoch = 10

        train_data, val_data = self.dataset.get_split(self.split)

        start = time.time()

        early_stopping_counter = 0

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
            print(f"Epoch {epoch + 1}/{self.num_epochs}\n")

            # set model to training mode
            self.model.train()

            # do a training epoch to get loss and acc, and record them
            train_loss, train_acc = self.training_epoch(train_data)
            stats['train_losses'].append(train_loss)
            stats['train_accs'].append(train_acc)

            print(f"Training loss - Epoch {epoch + 1}/{self.num_epochs}: {train_loss}")
            print(f"Training accuracy - Epoch {epoch + 1}/{self.num_epochs}: {train_acc}")

            # set model to training mode
            self.model.eval()

            # do a training epoch to get loss and acc, and record them
            val_loss, val_acc = self.validation_epoch(val_data)
            stats['val_losses'].append(train_loss)
            stats['val_accs'].append(train_acc)

            print(f"Validation loss - Epoch {epoch + 1}/{self.num_epochs}: {val_loss}")
            print(f"Validation accuracy - Epoch {epoch + 1}/{self.num_epochs}: {val_acc}")

            self.lr_scheduler.step()

            if val_acc > best_acc:
                # keep a copy of the best performing model parameters
                best_acc = val_acc
                best_model_params = copy.deepcopy(self.model.state_dict())

                # dont early stop if model still improving
                early_stopping_counter = 0
            else:

                # if validation accuracy goes down too much, model is overfitting so no point continuing
                early_stopping_counter += 1

            if early_stopping_counter == self.patience and self.early_stopping:
                stopped_early = True
                stopped_epoch = epoch

                break

        end = time.time() - start
        if stopped_early:
            print(f"Stopped training early after Epoch {stopped_epoch}")

        print(f"Training finished in {end:.0f}m {end:.0f}s")
        print(f"Best validation accuracy: {best_acc:4f}")

        # load best model weights
        self.model.load_state_dict(best_model_params)
        torch.save(self.model, f"{self.experiment_name}_{self.split}.pt")
