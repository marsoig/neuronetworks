import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers

def task5():
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
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        metrics = ["accuracy"],  # или другая метрика, см. https://www.tensorflow.org/api_docs/python/tf/keras/metrics
    )

    model.summary()

    dataset_raw = {
        (1, 20, 40, 20, 2, 1): 1,
        (0, 50, 54, 27, 2, 0): 0,
        (1, -2, 0, 20, 0, 0): 1,
        (1, -10, 40, 15, 2, 3): 2,
        (0, -10, 88, 25, 2, 0): 2,
        (4, 300, 0, -10, 1, 1): 1,
        (4, 120, 700, 15, 2, 1): 2,
        (2, -5, 10, 30, 2, 0): 0,
        (0, 300, 80, 100, 0, 0): 1,  # закрыть дверь, владелец уехал и забыл это сделать
        (1, 5, 30, 15, 0, 2): 2,
        (2, 20, 5, -10, 1, 3): 0,
        (3, 1, 4, 10, 2, 0): 2,
        (1, 0, 5, 25, 0, 1): 0,
        (1, 3, 10, 20, 2, 1): 1,
        (4, 0, 0, 20, 1, 0): 0,  # владелец мешает двери закрыться
        (1, -8, 4, 22, 2, 1): 1,
        (2, 10, 50, 251, 10, 3): 2,
        (0, 198, 999, 34, 2, 0): 0,
        (3, 267, 340, 27, 2, 4): 2,
        (3, 300, 100, 150, 2, 1): 1,
        (1, 40, 15, 40, 2, 1): 1,
        (1, 300, 0, 24, 1, 1): 2,
        (4, -8, -3, 10, 2, 3): 1,
        (1, -6, 3, 18, 0, 3): 0,
        (3, 198, 79, 14, 1, 0): 2,
        (4, 200, 0, -5, 1, 1): 1,
        (2, 250, 3, 90, 2, 0): 1,
        (0, 10, 15, -15, 2, 0): 0,
        (0, 20, 10, -20, 0, 1): 1,
        (0, 54, 20, 5, 2, 0): 0,
        (1, 20, 10, 5, 2, 1): 1,
        (2, -10, 0, 12, 2, 0): 2,
        (4, 0, 0, 10, 1, 0): 0,
        (2, 5, 8, 39, 2, 4): 2,
        (0, -10, 2, 15, 2, 0): 0,
        (1, 50, 50, 30, 1, 0): 0,
        (1, 200, 67, 19, 0, 1): 2,
        (1, 1, 3, 26, 2, 1): 1,
        (1, 246, 60, -21, 2, 3): 1,
        (0, 10, 5, -10, 0, 2): 0,
        (1, 3, 93, -10, 2, 0): 2,
        (2, -2, 0, 120, 2, 1): 0,
        (1, 299, 32, 20, 0, 1): 2,
        (0, 5, 3, 38, 1, 0): 0,  # игнорировать препятствия
        (2, 169, 121, 31, 1, 1): 1,
        (4, 100, 1000, 30, 2, 3): 2
    }
    DATASET_SIZE = len(dataset_raw)

    vectors = np.asarray([np.array(x) for x in dataset_raw.keys()])
    labels = np.fromiter(dataset_raw.values(), dtype=float)
    dataset = tf.data.Dataset.from_tensor_slices((tf.ragged.constant(vectors), labels)).shuffle(
        buffer_size=DATASET_SIZE)
    train_size = int(0.7 * DATASET_SIZE)
    test_size = int(0.3 * DATASET_SIZE)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size).take(test_size)
    #The right number of epochs depends on the inherent perplexity (or complexity) of your dataset.
    # A good rule of thumb is to start with a value that is 3 times the number of columns in your data.
    # If you find that the model is still improving after all epochs complete, try again with a higher value.
    # If you find that the model stopped improving way before the final epoch, try again with a lower value as you may be overtraining.

    model.fit(np.array(train_dataset), np.array(train_dataset), epochs=18, verbose=2)  # выберите число эпох

    model.evaluate(np.array(test_dataset), np.array(test_dataset), verbose=2)

    print(model.predict(np.array([[3, 120, 700, 15, 2, 1]])))

    # Что будете делать, если результат не 0, не 1 и не 2?
    # use the early stop approach based on the validation

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task5()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
