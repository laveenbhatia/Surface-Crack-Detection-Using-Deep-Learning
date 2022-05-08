import time
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.filters import sobel

NAME = f'CRACK-DETECTION-PRED-VGG16-{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')


def ext_features(img):
    sobel_img = sobel(img)
    return sobel_img


TRAIN_PATH = '../../data/Crack_Detect_Training'
train_batches = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1,
    preprocessing_function=ext_features
)
train_data = train_batches.flow_from_directory(
    directory=TRAIN_PATH,
    target_size=(224, 224),
    subset='training'
)

valid_data = train_batches.flow_from_directory(
    directory=TRAIN_PATH,
    target_size=(224, 224),
    subset='validation'
)

vgg = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
dense1 = Dense(128, input_shape=(224, 224, 3), activation='relu')(x)
dense2 = Dense(128, input_shape=(224, 224, 3), activation='relu')(dense1)
dense3 = Dense(128, input_shape=(224, 224, 3), activation='relu')(dense2)
dense4 = Dense(128, input_shape=(224, 224, 3), activation='relu')(dense3)
output = Dense(2, activation='softmax')(dense4)
model = Model(inputs=vgg.input, outputs=output)
print(model.summary())

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data,
          epochs=5,
          batch_size=32,
          validation_data=valid_data,
          callbacks=[tensorboard],
          validation_freq=[3, 5])

# model.save('model_surface_crack_pred_vgg16')

