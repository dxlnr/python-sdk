<h1 align="center">
  <b> Introduction using MNIST & Pytorch </b><br>
</h1>

Basic Federated Learning Example classifying Hand-written digits using the MNIST dataset. This setup is based on [Pytorch](https://pytorch.org/). This should serve as a basic introduction to the modalic platform. For further information visit our  [docs](https://modalic.ai/).

## Running the Setup

For aggregating the local models from different clients, modalic provides a lightweight server application which can be started by running
```shell
docker run modalic/worker:mosaic
```
Mosaic is the name of the aggregation server. The first time you run this command, the Mosaic docker image will be downloaded. This might take a bit of time and bandwidth, be patient. 
