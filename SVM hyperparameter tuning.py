# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:58:59 2021

@author: 36450057
"""

import xgboost as xgb
import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D,MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.utils import to_categorical
import keras
from skimage import io, filters, feature
from skimage.filters import sobel
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import VGG19
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import os
import seaborn as sns
import pandas as pd
import time
from time import sleep
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import xgboost

print(os.listdir('canola_training'))

#Resize images to
SIZE = 128


#creating empty lists.
train_images = []
train_labels = [] 

for directory_path in glob.glob('canola_training/training/*'):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.bmp")):
        print(img_path)
        orig_img   = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        orig_img   =cv2.resize(orig_img, (SIZE, SIZE)) 
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        train_images.append(orig_img)
        train_labels.append(label)
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)  



# test images and labels into respective lists
test_images = []
test_labels = [] 

for directory_path in glob.glob('canola_test/test/*'):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.bmp")):
        print(img_path)
        orig_img   = cv2.imread(img_path, cv2.IMREAD_COLOR)
        orig_img   =cv2.resize(orig_img, (SIZE, SIZE)) 
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        test_images.append(orig_img)
        test_labels.append(label)
        
test_images = np.array(test_images)
test_labels  = np.array(test_labels )  

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

###################################################################
# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0


#Load model wothout classifier/fully connected layers
VGG_model = VGG19(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

#Now, let us use features from convolutional network for RF
start_time = timer(None) # timing starts from this point for "start_time" variable
feature_extractor=VGG_model.predict(x_train)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)
X_for_training = features #This is our X input to RF
timer(start_time) # timing ends here for "start_time" variable


## Hyper Parameter Optimization

params = {'C': [1, 10, 20, 30, 40, 50],
          'gamma': [0.01, 0.001, 0.0001, 0.00001,0.000001],
          'kernel': ['rbf', 'poly', 'sigmoid']}


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
classifier=svm.SVC()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=10,n_jobs=-1,cv=6,verbose=3)

from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X_for_training,y_train)
timer(start_time) # timing ends here for "start_time" variable


random_search.best_estimator_
random_search.best_params_

#Classifier
SVM_model= svm.SVC(kernel='rbf', random_state=1, gamma=0.00001, C=32)


