import time
import os
import random
import tensorflow as tf
import numpy as np
import cv2
from skimage.filters import sobel
from sklearn.metrics import classification_report

model = tf.keras.models.load_model('model_surface_crack_pred_mlp')

DIRECTORY = "../../data/Crack_Detect_Testing"
CATEGORIES = ['Negative', 'Positive']
data = []
i = 0


# Data Preprocess
print("Preparing Testing Images...")
for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (128, 128))
        img_arr = sobel(img_arr)
        data.append([img_arr, label])
        i = i+1

print(i, "Testing images found")

random.shuffle(data)

X = []
y = []

for features, labels in data:
    X.append(features)
    y.append(labels)

x_test = np.array(X)
y_test = np.array(y)

x_test = x_test.reshape(-1, 49152)
x_test = x_test/255

# Load the model and make prediction on testing data
y_pred = model.predict(x_test, batch_size=32, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool, output_dict=True))

# eval_loss, eval_acc = model.evaluate(X, y)
#
# print('Evaluation Loss: {:.4f}, Evaluation Accuracy: {:.2f}'.format(eval_loss, eval_acc * 100))
