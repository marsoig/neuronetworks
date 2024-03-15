import pickle
import random

import PIL
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from keras.src.datasets import cifar10
from keras.models import load_model
from PIL import Image
from keras.callbacks import History
from tensorflow.python.keras.utils.np_utils import to_categorical

def task9():
    # Set random seeds for reproducibility
    seed_value = 42
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)

    # Define class names
    class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    input_shape = (32, 32, 3)

    y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # One-hot encode labels
    plt.imshow(x_train[99][:, :, 0])
    print(y_train[99])
    plt.show()

    batch_size = 32
    num_classes = 10
    epochs = 50

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    learning_rate = 0.00015  # Set your desired learning rate

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.995):
                print("\nReached 99.5% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()
    history= History()
    if (os.path.exists('cifar.h5')):
        model = load_model('cifar.h5')
        with open('history_cifar1.pkl', 'rb') as file:
            history.history = pickle.load(file)
    else:
        history = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.1,
                            callbacks=[callbacks])
        model.save('cifar.h5')

        # Save the history object separately
        with open('history_cifar1.pkl', 'wb') as file:
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
    # acc: 0.082

    # confusion matrix
    # Predict the values from the testing dataset
    Y_pred = model.predict(x_test)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    # Convert testing observations to one hot vectors
    Y_true = np.argmax(y_test, axis=1)

    # Compute the confusion matrix with specified labels
    confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes, num_classes=len(class_names))

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    # Load and preprocess a custom image
    custom_image_path = 'plane.jpg'
    image = Image.open(custom_image_path)
    image = image.resize((32, 32))

    # Display the reshaped image
    plt.imshow(image)  # Assuming it's grayscale
    plt.title('Custom Image')
    plt.show()

    # Make a prediction
    prediction = model.predict(np.array([image]))

    # Output the predicted class
    predicted_class = np.argmax(prediction)
    print(f'Predicted Class: {class_names[predicted_class]}')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task9()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
