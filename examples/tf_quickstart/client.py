"""Client Implementation"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import modalic


class TFFLClient(modalic.Client):
    r"""Client object implementing the Machine Learning logic using tensorflow."""

    def __init__(self):
        # Set important hyperparameters.
        self.batch_size = 32
        self.epochs = 5

        # Load the CIFAR-10 dataset using tf.keras.
        (self.x_train, self.y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

        # Initialize & compile the MobileNetV2 model.
        self.model = tf.keras.applications.MobileNetV2(
            (32, 32, 3), classes=10, weights=None
        )
        self.model.compile(
            "adam", "sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    def train(self, model):
        if model is not None:
            self.model = model

        self.model.fit(
            self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs
        )
        return self.model

    def serialize_local_model(self, model):
        return modalic.serialize_tf_keras_model(model)

    def deserialize_global_model(self, global_model):
        self.model = modalic.deserialize_tf_keras_model(
            self.model, global_model, self._get_model_shape()
        )

    def get_model_shape(self):
        return modalic.get_tf_keras_model_shape(self.model)

    def get_model_dtype(self):
        pass


def main():
    print("Preparing Federated Learning Client.\n")
    tffl_client = TFFLClient()

    modalic.run_client(tffl_client)


if __name__ == "__main__":
    main()
