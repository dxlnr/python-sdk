import torch
import torch.nn as nn
from torchvision import datasets, transforms

import modalic

server_address = "127.0.0.1:8080"
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


def load_data():
    r"""loading the mnist datasets."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    mnist_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return mnist_trainset, mnist_testset


class CNN(nn.Module):
    r"""simple 2D CNN model for classification."""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


class Trainer(object):
    r"""Trainer class object to perform the Learning.
    Args:
        device (torch.device): model running device. GPUs are recommended for model training and inference.
        dataset: (lib.data.data.Dataloader) Dataloader object.
    """

    def __init__(
        self, device: torch.device, dataset, epochs: int,
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

    def train(self, cid=0):
        self.model.train()

        running_loss = 0.0
        for epoch in range(0, self.epochs):
            for i, (images, labels) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output, x = self.model.forward(images)

                loss = self.loss(output, labels.long())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if (i + 1) == len(self.trainloader):
                    print(
                        f"[client {cid}, epoch {epoch + 1}, {i + 1:5d}] loss: {running_loss / len(self.trainloader):.3f}"
                    )

        return self.model, (running_loss / len(self.trainloader))


################################################################################
trainset, testset = load_data()

client = modalic.Client(Trainer(device, trainset, epochs=1), 1, server_address)
client.run()
