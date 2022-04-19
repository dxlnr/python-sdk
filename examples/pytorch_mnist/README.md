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

### Server
For aggregating the local models from different clients, modalic provides a lightweight server application which can be started by running
```
docker run modalic/worker:mosaic
```
Mosaic is the name of the aggregation server. The first time you run this command, the Mosaic docker image will be downloaded. This might take a bit of time and bandwidth, be patient.

### Clients
Installing all the dependencies by running
```
python3 -m venv modalic-env
source /modalic-env/bin/activate
pip install --editable .
```
An individual client can be started simply by running
```
python client.py
```
in an additional terminal.
