import pickle
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tensorflow import keras
from keras.models import load_model
from keras import layers
from PIL import Image
from keras.callbacks import History
from tensorflow.python.keras import Model


def task14():
    # Set random seeds for reproducibility
    seed_value = 42
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)

    mnist = tf.keras.datasets.mnist
    # The x_* datasets contain respectively 60000 and 10000 matrices of 28*28 pixels encoded as ints between 0 and 255.
    # The y_* datasets contain the labels of what number is represented in your corresponding 28*28 pixels matrices.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    input_shape = (28, 28, 1)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_train = x_train / 255.0
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_test = x_test / 255.0

    y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

    batch_size = 48
    num_classes = 10
    epochs = 6

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    learning_rate = 0.0015  # Set your desired learning rate

    # Create an RMSprop optimizer with the specified learning rate
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, epsilon=1e-08)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.995):
                print("\nReached 99.5% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()
    history= History()
    if (os.path.exists('ziffern.h5')):
        model = load_model('ziffern.h5')
        with open('history.pkl', 'rb') as file:
            history.history = pickle.load(file)
    else:
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.1,
                            callbacks=[callbacks])
        model.save('ziffern.h5')

        # Save the history object separately
        with open('history.pkl', 'wb') as file:
            pickle.dump(history.history, file)

    plt.imshow(x_train[0])
    plt.show()

    # Extract features after the second Conv2D layer
    layer_name = 'conv2d_1'  # Use the correct name for the second Conv2D layer
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(x_train[:1])

    # Plot the features (feature maps)
    num_filters = intermediate_output.shape[-1]
    plt.figure(figsize=(8, 8))
    for i in range(num_filters):
        plt.subplot(num_filters // 8, 8, i + 1)
        plt.imshow(intermediate_output[0, :, :, i], cmap='viridis')
        plt.axis('off')

    plt.show()

    # Create a model that outputs the activations of the MaxPooling2D layer
    model_intermediate = tf.keras.Model(inputs=model.input, outputs=model.layers[2].output)
    feature_maps = model_intermediate.predict(x_train[:1])

    # Display the feature maps
    num_features = feature_maps.shape[-1]
    fig, axs = plt.subplots(1, num_features, figsize=(15, 15))

    for i in range(num_features):
        axs[i].imshow(feature_maps[0, :, :, i], cmap='gray')
        axs[i].axis('off')

    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task14()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
