<h1 align="center">
  <b> Introduction using MNIST & Pytorch </b><br>
</h1>

Basic Federated Learning example classifying hand-written digits using the MNIST dataset. This setup is based on [Pytorch](https://pytorch.org/). This should serve as a basic introduction to the modalic platform. For further information visit our  [docs](https://modalic.ai/).

## Running the Setup

### Example setup
You can clone this example as a closed folder into a separate environment by running
```
git clone --depth=1 https://github.com/modalic/python-sdk.git && mv python-sdk/examples/pytorch_mnist . && rm -rf python_sdk && cd pytorch_mnist
```
Installing all the dependencies by running
```
python3 -m venv modalic-env
source modalic-env/bin/activate
pip install --editable .
```

### Download the dataset
Get the MNIST dataset via
```shell
mkdir -p data/MNIST/ && cd data/MNIST/ && wget https://data.deepai.org/mnist.zip && unzip mnist.zip -d mnist && rm mnist.zip && cd ../..
```

### Server
For aggregating the local models from different clients, modalic provides a lightweight server application which can be started by running
```
python server.py --cfg config.toml
```

### Clients
An individual client can be started simply by running
```shell
python client.py --client_id 1
# Additional clients can be started with a different client_id.
python client.py --client_id 2
# ...
```
in additional terminals.
