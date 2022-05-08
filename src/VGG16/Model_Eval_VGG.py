import tensorflow as tf
import os
import numpy as np
import cv2
import random
from skimage.filters import sobel
from sklearn.metrics import classification_report


model = tf.keras.models.load_model('model_surface_crack_pred_vgg16')

DIRECTORY = "../../data/Crack_Detect_Testing"
CATEGORIES = ['Negative', 'Positive']
data = []
i = 0
for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        img_arr = img_arr/255
        img_arr = sobel(img_arr)
        data.append([img_arr, label])
        #print(i)
        i = i+1

random.shuffle(data)

X = []
y = []

for features, labels in data:
    X.append(features)
    y.append(labels)

x_test = np.array(X)
y_test = np.array(y)

y_pred = model.predict(x_test, batch_size=32, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool, output_dict=True))

# eval_loss, eval_acc = model.evaluate(test_data)
#
# print('Evaluation Loss: {:.4f}, Evaluation Accuracy: {:.2f}'.format(eval_loss, eval_acc * 100))
