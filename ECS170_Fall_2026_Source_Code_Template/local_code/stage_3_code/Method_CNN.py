from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np

class Method_CNN(method, nn.Module):
    def __init__(self, mName,
                 mDescription,
                 input_channels=1, num_classes=10):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv_layer_1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=5,
            padding=2,
        )
        self.activation_func_1 = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.conv_layer_2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            padding=2,
        )
        self.activation_func_2 = nn.ReLU()

        self.pool_2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            ceil_mode=True,
        )

        self.fc_layer_1 = nn.Linear(32 * 7 * 7, 128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.activation_func_3 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(128, num_classes)

        self.max_epoch = 30
        self.learning_rate = 1e-3

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.to(self.device)

        print("Using device:", self.device)

    def forward(self, x):
        # First convolution
        x = self.activation_func_1(self.conv_layer_1(x))
        x = self.pool(x)

        #second convolution
        x = self.activation_func_2(self.conv_layer_2(x))

        x = self.pool_2(x)

        #Flatten
        x = self.adaptive_pool(x.cpu()).to(self.device)
        x = x.reshape(x.size(0), -1)

        #Fully Connected Layers
        x = self.activation_func_3(self.fc_layer_1(x))
        y_pred = self.fc_layer_2(x)

        return y_pred
    def fit(self, X, y):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.001
        )

        loss_function = nn.CrossEntropyLoss()

        X = torch.tensor(
            np.array(X), dtype=torch.float32
        )
        Y = torch.tensor(
            np.array(y), dtype=torch.long
        )

        batch_size = 128

        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        self.loss_history =[]
        self.accuracy_history =[]

        for epoch in range(self.max_epoch):
            total_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.forward(X_batch)

                train_loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                total_loss += train_loss.item() * X_batch.size(0)
                correct +=(y_pred.argmax(dim=1) == y_batch).sum().item()
                total += y_batch.size(0)

            epoch_loss = total_loss / total
            epoch_accuracy = correct / total

            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_accuracy)

            print(
                'Epoch:', epoch + 1,
                'Accuracy:', epoch_accuracy,
                'Loss:', epoch_loss,
            )
    def test(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred = self.forward(X)
            pred_y = torch.argmax(y_pred, dim=1)

        return pred_y.cpu()

    def run(self):
        import matplotlib.pyplot as plt
        print("Training CNN...")

        self.fit(
            self.data['train']['X'],
            self.data['train']['y'],
        )

        plt.figure()
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("CNN Training Loss")
        plt.savefig("cifar_kernel_5_training_loss.png")
        plt.show()

        plt.figure()
        plt.plot(range(1, len(self.accuracy_history) + 1), self.accuracy_history)
        plt.xlabel("Epoch")
        plt.ylabel("Training Accuracy")
        plt.title("CNN Training Accuracy")
        plt.savefig("cifar_kernel_5_training_accuracy.png")
        plt.show()


        torch.save(self.state_dict(), "cifar_kernel_5_cnn.pth")
        print("Model Saved")

        print("Testing CNN...")

        pred_y = self.test(self.data['test']['X'])

        test_accuracy = (
            pred_y.numpy() == self.data['test']['y']
        ).mean()

        print("Test Accuracy:", test_accuracy)

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
        }