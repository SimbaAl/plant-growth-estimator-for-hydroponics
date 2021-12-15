# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 05:37:36 2021

@author: 36450057
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:58:59 2021

@author: 36450057
"""
import graphviz
import xgboost
from xgboost import plot_tree
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
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
from tensorflow.keras.applications.resnet50 import ResNet50
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



print(os.listdir('canola_training'))

#Resize images to
SIZE = 224


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
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

for layer in resnet_model.layers:
	layer.trainable = False
    
resnet_model.summary()  #Trainable parameters will be 0

#Now, let us use features from convolutional network for RF
from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
feature_extractor=resnet_model.predict(x_train)
timer(start_time) 


features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_training = features #This is our X input to RF

## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.25, 0.30, 0.4, 0.5, 0.7] ,
 "max_depth"        : [5, 6, 8, 10, 12],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.4, 0.5, 0.6, 0.7 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=3,n_jobs=-1,cv=5,verbose=3)

from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X_for_training,y_train)
timer(start_time) # timing ends here for "start_time" variable


random_search.best_estimator_
random_search.best_params_


classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4,
              enable_categorical=False, gamma=0.5, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.25, max_delta_step=0, max_depth=6,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)


