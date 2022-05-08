import numpy as np
import tensorflow as tf
import cv2
from skimage.filters import sobel

model = tf.keras.models.load_model('model_surface_crack_pred_mlp')
CATEGORIES = ['negative', 'positive']

img_path_test = input("Please enter the path of the image: ")
img_arr_test = cv2.imread(img_path_test)
img_arr_test = cv2.resize(img_arr_test, (128, 128))
img_arr_test = sobel(img_arr_test)
img_arr_test = img_arr_test.reshape(-1, 49152)
img_arr_test = img_arr_test/255

prediction = model.predict(img_arr_test)
prediction = np.argmax(prediction)

if CATEGORIES[prediction] == 'negative':
    print('Prediction: Crack not detected')
else:
    print('Prediction: Crack detected')
