.. _quickstart_pytorch:

Quickstart PyTorch
==================

This example aims for a quick start into the functionality modalic core provides, by walking through a basic example.
The Federated Learning example will classify hand-written digits using the well-known `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset. 
The setup is based on `Pytorch <https://pytorch.org/>`_.

The example can be run as a standalone project via::

    git clone --depth=1 https://github.com/modalic/python-sdk.git && mv python-sdk/examples/pytorch_mnist . && rm -rf python_sdk && cd pytorch_mnist

And subsequently installing all the dependencies is done by running::
    
    python3 -m venv modalic-env
    source modalic-env/bin/activate
    pip install --editable .

Dataset
-------

The MNIST dataset can be downloaded and packaged into the correct folder structure via::

    mkdir -p data/MNIST/ && cd data/MNIST/ && wget https://data.deepai.org/mnist.zip && unzip mnist.zip -d mnist && rm mnist.zip && cd ../..

At this point, the setup is ready. Let's dive into the code!

For data preparation done in :code:`data.py`, a function is provided that slices up the data in chunks of certain size.
The parameter *num_splits* lets you control the number of chunks that should be sliced. A high number leads to small dataset size for
each individual client.

.. code-block:: python

    def load_partition_data_mnist(num_splits: int = 10) -> Tuple[Any, Any, Any, Any]:
    r"""partition training set into same sized splits."""
        dir = os.path.abspath("data/MNIST/mnist/")
        train_data, test_data, train_labels, test_labels = read_from_path(dir)
        return (
            np.split(train_data, num_splits),
            np.split(train_labels, num_splits),
            test_data,
            test_labels,
        )

Additionally, a custom Dataloader object in the spirit of Pytorch is implemented for handling the dataset during training.

.. code-block:: python

    class Dataloader(torch.utils.data.Dataset):
        r"""Dataloader class object."""

        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            return (self.data[idx], self.labels[idx])


Aggregation Server
------------------

The modalic core provides a server executable which can also be started via Python script.

The :code:`server.py` file contains the logic to start the server.

.. code-block:: python

    import argparse
    import modalic

    parser = argparse.ArgumentParser(description="Server arguments.")
    parser.add_argument("--cfg", type=str, help="configuration file (path)")

    args = parser.parse_args()

Besides the import of some dependencies, this allows for adding a config file via commandline for controlling important hyperparameter
for the Federated Learning process.


The server itself is then started by adding the following line:

.. code-block:: python

    modalic.run_server(args.cfg)



Client Side
-----------

The individual logic for the client is implemented in the :code:`client.py` file. Besides some necessary dependencies, 
there is the possiblity to alter the client ID via commandline argument using a parser.

.. code-block:: python

    import argparse
    import sys

    import torch
    import torch.nn as nn
    from data import Dataloader, load_partition_data_mnist

    import modalic


    def create_arg_parser():
        r"""Get arguments from command lines."""
        parser = argparse.ArgumentParser(description="Client parser.")
        parser.add_argument(
            "--client_id", metavar="N", type=int, help="an integer specifing the client ID."
        )

        return parser

The Deep Learning model architecture is defined as

.. code-block:: python

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

And as the last element for performing the training is the Trainer object itself.

.. code-block:: python

    class Trainer(object):
    r"""Trainer class object to perform the Learning.

    :param device: (torch.device) Model running device. GPUs are recommended for model training and inference.
    :param dataset: (data.Dataloader) Custom Dataloader object.
    :param epochs: (int) Epochs hyperparameter.
    """

    def __init__(
        self,
        device: torch.device,
        dataset: Dataloader,
        epochs: int = 1,
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
            for i, (images, labels) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output, _ = self.model.forward(images)

                loss = self.loss(output, labels.long())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

        return self.model, running_loss / len(self.trainloader)

Main Function
-------------

The last code block brings everything together and defines the the :code:`main` function:

.. code-block:: python

    def main():
        arg_parser = create_arg_parser()
        args = arg_parser.parse_args(sys.argv[1:])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_data, train_labels, test_data, test_labels = load_partition_data_mnist(
            num_splits=100
        )

        client = modalic.PytorchClient(
            Trainer(
                device,
                Dataloader(
                    train_data[args.client_id - 1], train_labels[args.client_id - 1]
                ),
            ),
            conf={
                "api": {"server_address": "[::]:8080"},
                "process": {"training_rounds": 10, "timeout": 5.0},
            },
            args.client_id,
        )
        client.train()

In the style of object-oriented programming, the modalic `PytorchClient <modalic-pytorch-client>`_ is used. The client implements
all the logic which is necessary to perform training in a federated fashion. The client contains a (custom) trainer object which has to implement a :code:`train()`
function. In addition, all the necessary hyperparameter are set via custom `Configuration <modalic-conf-apiref>`_ object.

Run the Training
----------------

As everything is setup, the training can start by running the aggregation server via

.. code-block:: shell

    $ python server.py --cfg config.toml

followed by starting the first client in a new terminal window via:

.. code-block:: shell

    $ python client.py --client_id 1

Start additional clients accordingly:

.. code-block:: shell

    $ python client.py --client_id 2

Start as many clients in order to match the overall number of participants stated in the config file.



