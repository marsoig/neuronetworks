import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers

def task4():
    # Feedforward artificial neural network models are mainly used for simplistic classification problems
    # fully-connected, feedforward Neural Network
    model = keras.Sequential(
        [
            keras.Input(shape=(6)),
            layers.Dense(6),
            # The dense layer is the fully connected, feedforward layer of a neural network.
            layers.Dense(3),
            layers.Dense(1),
        ]
    )

    model.compile(
        # Cross-entropy is the default loss function to use for multi-class classification problems.
        loss=keras.losses.categorical_crossentropy,
        # In most optimizers in Keras, the default learning rate value is 0.001
        # A good rule of thumb is to set the initial learning rate to a small value and then gradually increase it until convergence
        optimizer = keras.optimizers.Adam(learning_rate=0.001),  # подберите learning rate
        metrics = ["accuracy"],  # или другая метрика, см. https://www.tensorflow.org/api_docs/python/tf/keras/metrics
    )

    model.summary()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task4()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
