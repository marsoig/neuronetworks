import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.callbacks import History
from keras.models import load_model
from keras.src.datasets import imdb


def task15():
    # Set random seeds for reproducibility
    seed_value = 42
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)

    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)

    # dataset analysis

    print("The output categories are", np.unique(targets))
    print("The number of unique words is", len(np.unique(np.hstack(data))))

    length = [len(i) for i in data]
    print("The Average Review length is", np.mean(length))

    print("Label:", targets[0])
    print(data[0])

    index = imdb.get_word_index()
    reverse_index = dict([(value, key) for (key, value) in index.items()])
    decoded = " ".join([reverse_index.get(i - 3, "#") for i in data[0]])
    print(decoded)

    # data preparation for training

    def vectorize(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1
        return results

    data = vectorize(data)
    targets = np.array(targets).astype("float32")

    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[40000:]
    train_y = targets[40000:]

    batch_size = 80
    epochs=200

    # Define the maximum number of words in your vocabulary
    vocab_size = 10000

    # Define the maximum length of a review
    max_len = 100

    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=50, input_length=max_len),
        tf.keras.layers.LSTM(units=100),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Use 'sigmoid' for binary classification
    ])

    model.summary()

    learning_rate = 0.0001  # Set your desired learning rate

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.995):
                print("\nReached 99.5% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()
    history= History()
    if (os.path.exists('imdb.h5')):
        model = load_model('imdb.h5')
        with open('history_imdb.pkl', 'rb') as file:
            history.history = pickle.load(file)
    else:
        history = model.fit(train_x, train_y,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(test_x, test_y),
                            callbacks=[callbacks])
        model.save('imdb.h5')

        # Save the history object separately
        with open('history_imdb.pkl', 'wb') as file:
            pickle.dump(history.history, file)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training Loss")
    ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss")
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation Accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.show()

    model.evaluate(test_x, test_y)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task15()
