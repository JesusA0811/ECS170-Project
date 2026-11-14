'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np

class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # Each image has 784 pixels (28 x 28); the hidden layer has 256 neurons.
        self.fc_layer_1 = nn.Linear(784, 256)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        # Output one score for each digit, from 0 to 9.
        self.fc_layer_2 = nn.Linear(256, 10)

        # return logits directly for nn.CrossEntropyLoss

        # use the GPU when available, otherwise use the CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

        print("Using device:", self.device)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        # outout layer result
        # self.fc_layer_2(h) will be a nx10 tensor
        # n (denotes the input instance number): 0th dimension; 10 (denotes the class number): 1st dimension
        # return logits; nn.CrossEntropyLoss handles the normalization during training
        y_pred = self.fc_layer_2(h)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()

        # get the input, we need to covert X into torch.tensor so pytorch algorithm can operate on it
        # Normalize pixel values from 0-255 to 0-1
        X = torch.tensor(
            np.array(X), dtype=torch.float32
        ) / 255.0

        # convert y to torch.tensor as well
        y = torch.tensor(np.array(y), dtype=torch.long)

        # Create mini-batches
        batch_size = 128
        # Keep each image paired with its correct label.
        dataset = torch.utils.data.TensorDataset(X, y)
        # Shuffle the training examples each epoch and train on up to 128 at a time.
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Store loss for our convergence plot
        self.loss_history = []

        # it will be an iterative gradient updating process
        # split X and y into smaller-sized batches
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...

            total_loss = 0.0
            correct = 0
            total = 0

            # Process one mini-batch at a time
            for X_batch, y_batch in dataloader:
                # Move the batch to the selected device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward propagation
                y_pred = self.forward(X_batch)

                # calculate the training loss
                train_loss = loss_function(y_pred, y_batch)

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

                # Track training performance
                # Weight each batch loss by its size so the smaller last batch counts correctly.
                total_loss += train_loss.item() * X_batch.size(0)
                # Count predictions whose highest-scoring digit matches the label.
                correct += (y_pred.argmax(dim=1) == y_batch).sum().item()
                total += y_batch.size(0)

            # Average loss and accuracy for this epoch
            epoch_loss = total_loss / total
            epoch_accuracy = correct / total

            self.loss_history.append(epoch_loss)

            print('Epoch:', epoch + 1, 'Accuracy:', epoch_accuracy, 'Loss:', epoch_loss)

    def test(self, X):
        # do the testing, and result the result
        X = torch.tensor(
            np.array(X), dtype=torch.float32
        ) / 255.0

        X = X.to(self.device)

        # Testing needs predictions only
        with torch.no_grad():
            y_pred = self.forward(X)

        # convert the output scores to the corresponding labels
        # instances will get the labels corresponding to the largest score
        # Move predicted labels to the CPU for evaluation and saving.
        return y_pred.argmax(dim=1).cpu()
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
