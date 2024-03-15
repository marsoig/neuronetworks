import os
import pickle
import random

import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import History
import numpy as np
import tensorflow as tf
from PIL import Image

def load_and_preprocess_images(folder_path, target_size=(32, 32)):
    images = []
    labels = []

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            # Read and resize the image using PIL
            image = Image.open(image_path)
            image = image.resize(target_size)
            # Convert to numpy array and normalize
            image = np.array(image).astype("float32") / 255.0
            images.append(image)
            labels.append(1)  # Label 1 for minivan images

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def get_dataset():
    # Set paths and parameters
    minivan_folder = 'dataset/Minivan'
    num_samples = 806
    target_size = (32, 32)

    # Load minivan images
    minivan_images, minivan_labels = load_and_preprocess_images(minivan_folder, target_size)

    # Load CIFAR-10 images and labels
    cifar10_dataset = tf.keras.datasets.cifar10
    (_, _), (cifar10_images, cifar10_labels) = cifar10_dataset.load_data()

    cifar10_images = cifar10_images.astype("float32") / 255.0

    # Ensure there are not more than 750 CIFAR-10 images
    if len(cifar10_images) > num_samples:
        indices = np.random.choice(len(cifar10_images), num_samples, replace=False)
        cifar10_images = cifar10_images[indices]
        cifar10_labels = cifar10_labels[indices]

    # Assign label 0 to CIFAR-10 images
    cifar10_labels = np.zeros(len(cifar10_images))

    # Combine minivan and CIFAR-10 images
    all_images = np.concatenate([minivan_images, cifar10_images], axis=0)
    all_labels = np.concatenate([minivan_labels, cifar10_labels], axis=0)

    # Shuffle the dataset
    indices = np.arange(len(all_images))
    np.random.shuffle(indices)

    all_images = all_images[indices]
    all_labels = all_labels[indices]
    return all_images, all_labels

def task11():
    # Set random seeds for reproducibility
    seed_value = 42
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)

    all_images, all_labels = get_dataset()
    input_shape = (32, 32, 3)

    # Convert NumPy arrays to TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))

    # Shuffle and split the dataset into training and test sets
    dataset = dataset.shuffle(len(all_images), reshuffle_each_iteration=False)

    # Calculate the number of samples for the test set
    num_test_samples = int(0.1 * len(all_images))
    print(num_test_samples)

    # Create training and test sets
    train_dataset = dataset.skip(num_test_samples)
    test_dataset = dataset.take(num_test_samples)

    # Convert the datasets to NumPy arrays for convenience
    def dataset_to_numpy(dataset):
        images, labels = [], []
        for img, label in dataset:
            images.append(img.numpy())
            labels.append(label.numpy())
        return np.array(images), np.array(labels)

    # Get NumPy arrays for training and test sets
    x_train, y_train = dataset_to_numpy(train_dataset)
    x_test, y_test = dataset_to_numpy(test_dataset)

    print(len(x_test))

    plt.imshow(x_train[1][:, :, 0])
    print(y_train[1])
    # plt.show()

    plt.imshow(x_train[2][:, :, 0])
    print(y_train[2])
    # plt.show()

    plt.imshow(x_train[3][:, :, 0])
    print(y_train[3])
    # plt.show()

    plt.imshow(x_train[4][:, :, 0])
    print(y_train[4])
    # plt.show()

    batch_size = 8
    epochs=40

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Use 'sigmoid' for binary classification
    ])

    learning_rate = 0.0001  # Set your desired learning rate

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.995):
                print("\nReached 99.5% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()
    history= History()
    if (os.path.exists('minivan.h5')):
        model = load_model('minivan.h5')
        with open('history_minivan.pkl', 'rb') as file:
            history.history = pickle.load(file)
    else:
        history = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            callbacks=[callbacks])
        model.save('minivan.h5')

        # Save the history object separately
        with open('history_minivan.pkl', 'wb') as file:
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

    # confusion matrix
    # Predict the values from the testing dataset
    Y_pred = model.predict(x_test)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred)
    # Convert testing observations to one hot vectors
    Y_true = np.argmax(y_test)

    # Load and preprocess a custom image
    custom_image_path = 'flower.jpg'
    image = Image.open(custom_image_path)
    image = image.resize((32, 32))

    image_array = np.array(image).astype("float32") / 255.0  # Normalize to [0, 1]

    # Display the reshaped image
    plt.imshow(image)
    plt.title('Custom Image')
    plt.show()

    # Make a prediction
    prediction = model.predict(np.array([image]))
    print(prediction)

    if (np.round(prediction) == 1):
        print("Yes")
    else:
        print("No")

    # Load and preprocess a custom image
    custom_image_path = 'minivan.png'
    image = Image.open(custom_image_path)
    image = image.convert('RGB')
    image = image.resize((32, 32))

    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = image_array.reshape((32, 32, 3))

    # Display the reshaped image
    plt.imshow(image_array)
    plt.title('Custom Image')
    plt.show()

    # Make a prediction
    prediction = model.predict(np.array([image_array]))
    print(prediction)

    if (np.round(prediction) == 1):
        print("Yes")
    else:
        print("No")

    # Load and preprocess a custom image
    custom_image_path = 'photo_2023-12-25_13-33-22.jpg'
    image = Image.open(custom_image_path)
    image = image.resize((32, 32))

    image_array = np.array(image).astype("float32") / 255.0  # Normalize to [0, 1]

    # Display the reshaped image
    plt.imshow(image)
    plt.title('Custom Image')
    plt.show()

    # Make a prediction
    prediction = model.predict(np.array([image]))
    print(prediction)

    if (np.round(prediction) == 1):
        print("Yes")
    else:
        print("No")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task11()
