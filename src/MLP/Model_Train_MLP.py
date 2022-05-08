import time
import os
import random
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from skimage.filters import sobel


# This function will preprocess the training data
# -----------------------------------------------
# Arguments:
#   path - this is the path of the folder, relative to this file which contains training images
def preprocess_training_data(path):
    # Classification Categories
    # Negative - Crack not detected
    # Positive - Crack detected
    CATEGORIES = ['Negative', 'Positive']
    data = []
    number_of_images = 0
    print("Preparing Training Images...")
    # Looping through each category
    for category in CATEGORIES:
        folder = os.path.join(path, category)
        label = CATEGORIES.index(category)
        # Looping through each image in each category,
        # applying the sobel filter,
        # resizing it and converting it to an array
        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            img_arr = cv2.imread(img_path)
            img_arr = cv2.resize(img_arr, (128, 128))
            img_arr = sobel(img_arr)
            data.append([img_arr, label])
            number_of_images += 1
    print(number_of_images, "Training images found")
    random.shuffle(data)
    X = []
    y = []
    # Creating the vectors for features and labels
    # X - features vector
    # y - labels vector
    for features, labels in data:
        X.append(features)
        y.append(labels)

    X = np.array(X)
    y = np.array(y)

    # Reshaping it to a 2D numpy array to be used as input for MLP
    X = X.reshape(-1, 49152)

    # Image normalization
    X = X/255
    return X, y


# This function is used to define the model
# Here we can play around with following hyper-parameters:
# 1. Add or delete layers
# 2. Change number of neurons for each layer
# 3. Change the activation function
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.build((None, 49152))
    return model


# This function will compile and fit the model
def train_model(model):
    # Compiling the model
    # Here we can change following hyper-parameters:
    #   1. optimizers
    #   2. loss function
    #   3. metrics
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Fitting the model
    # Here we can change following hyper-parameters:
    #   1. number of epochs
    #   2. batch size
    model.fit(X_train,
              y_train,
              epochs=50,
              validation_split=0.1,
              batch_size=32,
              callbacks=[tensorboard])

    return model


# This variable stores the name of the folder with timestamp which will store the logs of the model
NAME = f'CRACK-DETECTION-PRED-MLP-{int(time.time())}'

# Used to store logs
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

# Path of the folder, relative to this file where training images are stored
TRAINING_IMAGES_DIRECTORY = "../../data/Crack_Detect_Training"

# Creating the X and y training vectors for our model
X_train, y_train = preprocess_training_data(TRAINING_IMAGES_DIRECTORY)

# Creating the model
model = create_model()
print(model.summary())

# Training the model
model = train_model(model)

# Saving the model in a h5py format
model.save('model_surface_crack_pred_mlp')

