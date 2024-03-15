import os
import pickle
import random
import glob

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from keras.callbacks import History
from keras.models import load_model
from keras.src.datasets import imdb
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split

import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_process_audio(file_path, target_duration=1):
    audio, sr = librosa.load(file_path, duration=target_duration, sr=22050)  # Adjust sr as needed
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512)  # Adjust parameters as needed
    return mfccs

# Load Data and Labels
def load_dataset(main_directory):
    data = []
    labels = []
    classes = os.listdir(main_directory)

    label_encoder = LabelEncoder()

    for class_name in classes:
        class_path = os.path.join(main_directory, class_name)
        for audio_file in glob.glob(os.path.join(class_path, '*.wav')):
            features = load_and_process_audio(audio_file)
            data.append(features)
            labels.append(class_name)

    # Convert class labels to integers
    integer_encoded = label_encoder.fit_transform(labels)

    # One-hot encode the integer labels using tf.one_hot
    onehot_encoded = tf.one_hot(integer_encoded, depth=len(classes))

    # Find the maximum length of inner lists
    max_length = max(len(inner_list) for sublist in data for inner_list in sublist)
    print(max_length)

    # Pad the inner lists to have consistent length
    padded_data = [
        pad_sequences(sublist, maxlen=max_length, padding='post', truncating='post', value=0) for sublist in data
    ]

    return np.array(padded_data), np.array(onehot_encoded)


def task():
    class_names = ["dogs", "people", "cats"]
    # Set random seeds for reproducibility
    seed_value = 42
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)

    data, labels = load_dataset('Audio')

    print(data.shape)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    batch_size = 32
    epochs=30

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(13, 44)),  # Adjust input shape based on your feature extraction
        # tf.keras.layers.Reshape((1, 13, 44)),
        tf.keras.layers.LSTM(13),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.summary()

    learning_rate = 0.001  # Set your desired learning rate

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.995):
                print("\nReached 99.5% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()
    history= History()
    if (os.path.exists('sounds.h5')):
        model = load_model('sounds.h5')
        with open('history_sounds.pkl', 'rb') as file:
            history.history = pickle.load(file)
    else:
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            callbacks=[callbacks])
        model.save('sounds.h5')

        # Save the history object separately
        with open('history_sounds.pkl', 'wb') as file:
            pickle.dump(history.history, file)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training Loss")
    ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss")
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation Accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.show()

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



    model.evaluate(x_test, y_test)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task()
