"""Client Implementation"""
import random

import torch
import torch.nn as nn
from data import Dataloader, load_partition_data_mnist

import modalic


class CNN(nn.Module):
    r"""simple 2D CNN model for classification."""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


class FLClient(modalic.Client):
    r"""Trainer class object to perform the Learning.

    :param device: (torch.device) Model running device.
        GPUs are recommended for model training and inference.
    :param dataset: (data.Dataloader) Custom Dataloader object.
    :param epochs: (int) Epochs hyperparameter.
    """

    def __init__(
        self,
        device: torch.device,
        dataset: Dataloader,
        epochs: int = 5,
    ):
        self.device = device
        self.dataset = dataset
        self.epochs = epochs

        self.trainloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=32, shuffle=True
        )

        self.model = CNN()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        self.model.train()

        running_loss = 0.0
        for epoch in range(0, self.epochs):
            for _, (images, labels) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output, _ = self.model.forward(images)

                loss = self.loss(output, labels.long())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(
                f"\t[epoch {epoch + 1}] loss: {running_loss / len(self.trainloader):.3f}"
            )

        return self.model

    def serialize_local_model(self, model):
        return modalic.serialize_torch_model(model)

    def deserialize_global_model(self, global_model):
        self.model = modalic.deserialize_torch_model(
            self.model, global_model, self._get_model_shape()
        )

    def get_model_shape(self):
        return modalic.get_torch_model_shape(self.model)

    def get_model_dtype(self):
        pass


def main():
    # Define the computational device for torch.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Preparing dataset.")
    train_data, train_labels, _, _ = load_partition_data_mnist(num_splits=25)

    print("Preparing Federated Learning Client.\n")
    client = FLClient(
        device,
        Dataloader(
            train_data[random.randint(0, 25)], train_labels[random.randint(0, 25)]
        ),
    )

    # Start the FL Client
    modalic.run_client(client)


if __name__ == "__main__":
    main()
