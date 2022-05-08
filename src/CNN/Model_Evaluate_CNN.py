import tensorflow as tf
import os
import numpy as np
import cv2
import random
from skimage.filters import sobel
from sklearn.metrics import classification_report


def data_preprocess(path):
    CATEGORIES = ['Negative', 'Positive']
    data = []
    i = 0
    for category in CATEGORIES:
        folder = os.path.join(path, category)
        label = CATEGORIES.index(category)
        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            img_arr = cv2.imread(img_path)
            img_arr = cv2.resize(img_arr, (128, 128))
            img_arr = img_arr / 255
            img_arr = sobel(img_arr)
            data.append([img_arr, label])
            i = i + 1
    random.shuffle(data)
    X = []
    y = []

    for features, labels in data:
        X.append(features)
        y.append(labels)

    x_test = np.array(X)
    y_test = np.array(y)
    return x_test, y_test


# Path to Testing images directory
DIRECTORY = "../../data/Crack_Detect_Testing"

# Generate x and y vector for testing data
# x_test - features
# y_test - labels
x_test, y_test = data_preprocess(DIRECTORY)

# Load the model
model = tf.keras.models.load_model('model_surface_crack_pred_cnn')

# Make prediction
y_pred = model.predict(x_test, batch_size=32, verbose=1)

# Get the boolean labels from prediction
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool, output_dict=True))

# eval_loss, eval_acc = model.evaluate(test_data)
#
# print('Evaluation Loss: {:.4f}, Evaluation Accuracy: {:.2f}'.format(eval_loss, eval_acc * 100))
