<h1 align="center">
  <b> NLP example using TorchText library & Pytorch in a functional manner</b><br>
</h1>

Federated Learning example classifying text. This setup is based on [Pytorch](https://pytorch.org/) and the [TorchText](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html) library. This introduces the possiblity of using modalic in a more functional manner via decorators. For further information visit our  [docs](https://modalic.ai/).

## Running the Setup

### Example setup
You can clone this example as a closed folder into a separate environment by running
```
git clone --depth=1 https://github.com/modalic/python-sdk.git && mv python-sdk/examples/pytorch_nlp_func . && rm -rf python_sdk && cd pytorch_nlp_func
```
Installing all the dependencies by running
```
python3 -m venv modalic-env
source modalic-env/bin/activate
pip install --editable .
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
