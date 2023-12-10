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

def task7():
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
    plt.imshow(x_train[99][:, :, 0])
    print(y_train[99])
    plt.show()

    batch_size = 32
    num_classes = 10
    epochs = 4

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

    learning_rate = 0.01  # Set your desired learning rate

    # Create an RMSprop optimizer with the specified learning rate
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, epsilon=1e-08)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.995):
                print("\nReached 99.5% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()
    history=None
    if (os.path.exists('ziffern.h5')):
        history = load_model('ziffern.h5')
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

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training Loss")
    ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss")
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation Accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.show()

    test_loss, test_acc = model.evaluate(x_test, y_test)
    # acc: 0.0965

    # confusion matrix
    # Predict the values from the testing dataset
    Y_pred = model.predict(x_test)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    # Convert testing observations to one hot vectors
    Y_true = np.argmax(y_test, axis=1)
    # compute the confusion matrix
    confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt='g')
    plt.show()

    # Load and preprocess a custom image
    custom_image_path = 'digit6.jpg'
    custom_image = Image.open(custom_image_path).convert('L')  # Convert to grayscale
    custom_image = custom_image.resize((28, 28))  # Resize to 28x28 pixels
    custom_image_array = np.array(custom_image) / 255.0  # Normalize to [0, 1]

    custom_image_reshaped = custom_image_array.reshape((28, 28, 1))

    # Display the reshaped image
    plt.imshow(custom_image_reshaped[:, :, 0], cmap='gray')  # Assuming it's grayscale
    plt.title('Custom Image')
    plt.show()

    # Make a prediction
    prediction = model.predict(np.array([custom_image_reshaped]))

    # Output the predicted class
    predicted_class = np.argmax(prediction)
    print(f'Predicted Class: {predicted_class}')
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task7()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
