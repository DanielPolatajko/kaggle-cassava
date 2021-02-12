# Utility class to build experiments, allowing easy changing of hyperparams, models, etc

import torch
import time
import copy

from tqdm import tqdm

from loss import BiTemperedLogisticLoss, SnapMix, SnapMixLoss
import torch.optim as optim
import numpy as np

class Experimenter:
    """
    Class to construct Pytorch experiment
    """

    def __init__(self, model, model_name, device, loss_function, optimiser, optimiser_config, lr_scheduler,
                 lr_config, dataset, data_batch_size, training_batch_size, num_epochs, num_classes, img_size,
                 warmup_scheduler=None, flatness=12, early_stopping=True, patience=3, tta=5, snapmix=None):
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
        self.init_model = copy.deepcopy(model.state_dict())
        self.model_name = model_name
        self.device = device
        self.loss_function = loss_function

        self.init_optimiser = optimiser
        self.optimiser_config = optimiser_config
        self.init_lr_scheduler = lr_scheduler
        self.lr_config = lr_config

        self.optimiser = optimiser
        self.lr_scheduler = lr_scheduler(self.optimiser, **self.lr_config)

        if warmup_scheduler is not None:
            self.warmup_scheduler = warmup_scheduler(optimiser)
        else:
            self.warmup_scheduler = None

        self.dataset = dataset
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.one_hot = False
        self.tta = tta
        self.num_classes = num_classes
        self.flatness = flatness
        self.flatness_counter = 0

        self.scaler = torch.cuda.amp.GradScaler()

        self.img_size = img_size

        if snapmix is None:
            self.snapmix = None
        else:
            self.snapmix = SnapMix()
            self.snapmix_prob = snapmix['prob']
            self.snapmix_alpha = snapmix['alpha']
            self.snapmix_loss = SnapMixLoss()

        self.grad_accum = training_batch_size // data_batch_size

        if type(self.loss_function) == BiTemperedLogisticLoss:
            print("one hot")
            self.one_hot = True

    def training_epoch(self, train_data):
        """
        Method to execute a training epoch
        :return:
        """

        # initiliase epoch loss and predictions for accuracy
        cumulative_loss = 0.0
        correct_preds = 0

        batch_loss = 0

        # zero the parameter gradients
        self.optimiser.zero_grad()

        with tqdm(train_data, unit="batch") as tepoch:
            for i, (inputs, labels) in enumerate(tepoch):

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                with torch.cuda.amp.autocast():


                    # Get model outputs and calculate loss
                    if self.one_hot:

                        one_hot_labels = torch.zeros(inputs.size(0), self.num_classes).cuda()
                        one_hot_labels[range(one_hot_labels.shape[0]), labels] = 1

                        if self.snapmix is not None:
                            # do the snapmix thing
                            rand = np.random.rand()
                            if rand > (1.0 - self.snapmix_prob):
                                X, ya, yb, lam_a, lam_b = self.snapmix(inputs, one_hot_labels, self.snapmix_alpha, self.img_size, self.model)
                                outputs, _ = self.model(X)
                                loss = self.snapmix_loss.forward(self.loss_function, outputs, ya, yb, lam_a, lam_b)

                            else:
                                outputs, _ = self.model(inputs)
                                loss = self.loss_function(outputs, one_hot_labels)
                        else:
                            outputs, _ = self.model(inputs)
                            loss = self.loss_function(outputs, one_hot_labels)

                    else:
                        outputs = self.model(inputs)
                        loss = self.loss_function(outputs, labels)

                    loss = loss / self.grad_accum

                #batch_loss += loss.item()

                self.scaler.scale(loss).backward()

                # make predictions
                _, preds = torch.max(outputs, 1)

                if (i + 1) % self.grad_accum == 0:  # Wait for several backward steps
                    # backward pass and parameter optimisation, with autoscaling
                    self.scaler.step(self.optimiser)
                    self.scaler.update()
                    # zero the parameter gradients
                    self.optimiser.zero_grad()

                    #cumulative_loss += batch_loss * inputs.size(0) * self.grad_accum
                    #tepoch.set_postfix(loss=batch_loss)
                    #batch_loss = 0.



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

                # forward pass
                with torch.set_grad_enabled(False):
                    # Get model outputs and calculate loss
                    if self.snapmix is not None:
                        outputs, _ = self.model(inputs)
                    else:
                        outputs = self.model(inputs)

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


        for fold in range(self.dataset.k):

            # allows us to check when early stopping kicked in
            stopped_early = False
            stopped_epoch = 10

            # instantiate and send the model to GPU
            self.model.load_state_dict(self.init_model)
            self.model = self.model.to(self.device)

            # initialise optimiser and lr scheduler
            #self.optimiser = self.init_optimiser(self.model.parameters(), **self.optimiser_config)
            #self.lr_scheduler = self.init_lr_scheduler(self.optimiser, **self.lr_config)


            # get the dataset split
            train_data, val_data = self.dataset.get_split(fold)


            # keep track of the time, and initiate an early stopping counter (which records how many epochs val acc
            # has not increased for)
            start = time.time()
            early_stopping_counter = 0

            # keep losses and accuracies to plot training afterwards
            stats = {
                'train_losses': [],
                'train_accs': [],
                'val_losses': [],
                'val_accs': []
            }

            # we will want to keep the best model
            best_model_params = copy.deepcopy(self.model.state_dict())
            best_acc = 0.0

            # iterate through the epochs
            for epoch in range(self.num_epochs):
                print(f"Epoch {epoch+1}/{self.num_epochs}\n")

                # set model to training mode
                self.model.train()

                # do a training epoch to get loss and acc, and record them
                train_loss, train_acc = self.training_epoch(train_data)
                stats['train_losses'].append(train_loss)
                stats['train_accs'].append(train_acc)

                print(f"Training loss - Epoch {epoch+1}/{self.num_epochs}: {train_loss}")
                print(f"Training accuracy - Epoch {epoch+1}/{self.num_epochs}: {train_acc}")

                # set model to training mode
                self.model.eval()

                # do a training epoch to get loss and acc, and record them
                val_loss, val_acc = self.validation_epoch(val_data)
                stats['val_losses'].append(train_loss)
                stats['val_accs'].append(train_acc)

                print(f"Validation loss - Epoch {epoch+1}/{self.num_epochs}: {val_loss}")
                print(f"Validation accuracy - Epoch {epoch+1}/{self.num_epochs}: {val_acc}")

                # learning rate scheduling
                if self.flatness_counter >= self.flatness:
                    self.lr_scheduler.step()
                    if self.warmup_scheduler is not None:
                        self.warmup_scheduler.dampen()

                # update the flatness counter to check whether to kick in the lr scheduler
                self.flatness_counter += 1

                if val_acc > best_acc:
                    # keep a copy of the best performing model parameters
                    best_acc = val_acc
                    best_model_params = copy.deepcopy(self.model.state_dict())

                    # dont early stop if model still improving
                    early_stopping_counter = 0
                else:

                    # if validation accuracy goes down too much, model is overfitting so no point continuing
                    early_stopping_counter += 1

                # if val acc has not increased for a few epochs, end the training loop
                if early_stopping_counter == self.patience and self.early_stopping:
                    stopped_early = True
                    stopped_epoch = epoch

                    break

            # print some final training stats
            end = time.time() - start
            if stopped_early:
                print(f"Stopped training early after Epoch {stopped_epoch}")

            print(f"Training finished in {end:.0f}m {end:.0f}s")
            print(f"Best validation accuracy: {best_acc:4f}")


            # save the best model from the training loop
            self.model.load_state_dict(best_model_params)
            torch.save(self.model, f"{self.model_name}_{fold}.pt")