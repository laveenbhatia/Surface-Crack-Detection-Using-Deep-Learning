# Surface Crack Detection using Deep Learning


## Introduction
Crack detection plays an important role in preventive maintenance of concrete structures. The manual inspection by humans can be time and resource intensive. Therefore, automatic crack detection using machine learning techniques has gained widespread attention recently. In this study, we have used the Surface Crack Detection dataset from kaggle and trained it on 3 different deep learning algorithms - Convolutional Neural Networks (CNN), Multilayer Perceptron (MLP) and Transfer Learning using VGG16. We also applied sobel edge detection filter to enhance the cracks during image pre-processing stage. We then compared the accuracy of above mentioned four models on the testing data.

## Dataset
The dataset used in this project is [Surface Crack Detection - Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) dataset, which is a collection of 40000 images divided into 2 classes - 
* Negative - Images with no crack
* Positive - Images with crack  
The image data can be found in the following subdirectory of this repo - main/data/

## Preprocessing the data
#### Data Normalization
To normalize the image data, the pixel values in image array are divided by 255 to have values between 0 and 1.
#### Data Augmentation
To perform data augmentation, following augmentation techniques are used:
* Rotation - randomly rotate images by 90 degrees
* Horizontal Flip - randomly flip images horizontally
* Vertical Flip - randomly flip images vertically
* Brightness Range - randomly brighten the images within a specified range
* Zoom Range - randomly perform zoom on images by a given factor
#### Image Filter
* We are using sobel image filter as our preprocessing feature extractor. The Sobel operator performs a 2-D spatial gradient measurement on an image and so emphasizes regions of high spatial frequency that correspond to edges.
## Training the model
To train the model, I have used the Sequential() class of tensorflow.keras library. In this sequential model, we add our layers as required.
The code for the training of the model can be found in - main/src/

## Evaluation the model
Evaluation of the model is done using the following matrices:
* Accuracy
* Precision
* Recall
* F1 Score
The code for the evaluation can be found in - main/src/

## Instructions to run the project
* Run the folloiwng command to clone the github repo
```bash
git clone https://github.com/laveenbhatia/Using-CNN-for-Sentiment-Analysis-of-Noisy-Audio-Data.git
```

* Run the following command to install the required dependencies
```bash
pip install -r requirements.txt
```

* If you wish to use your own custom data to train then add your custom images in the repective class folder in data/Crack_Detect_Training/[Negative/Positive]
* To train the model, go the the folder main/src. Here you will see 3 different folders - CNN, MLP, VGG16. Each folder caontains the source code to train the respective model. For example, the folder CNN contains a file Model_Train_CNN.py. Run this to train the CNN model. If required, the hyper parameters can be tuned in this file itself.
* Similarly to evaluate the model, go to main/src. Then go to the folder for which model you wish to evaluate and run Model_Evaluate.py.
