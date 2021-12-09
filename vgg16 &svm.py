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

#Now, let us use features from convolutional network for RF
feature_extractor=VGG_model.predict(x_train)

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
import xgboost
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=2,n_jobs=-1,cv=5,verbose=3)

from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X_for_training,y_train)
timer(start_time) # timing ends here for "start_time" variable


random_search.best_estimator_
random_search.best_params_


classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5,
              enable_categorical=False, gamma=0.4, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.4, max_delta_step=0, max_depth=8,
              min_child_weight=3, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

from sklearn.model_selection import cross_val_score
#score=cross_val_score(classifier,X_for_training,y_train,cv=3)
#print(score)
#score.mean()

import graphviz
from xgboost import plot_tree
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams

##set up the parameters
rcParams['figure.figsize'] = 150, 100
plot_tree(classifier,num_trees=2)

#Send test data through same feature extractor process
start_time = timer(None)
X_test_feature = VGG_model.predict(x_test)
timer(start_time) # timing ends here for "start_time" variable
#X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)


#Extract features from test data and reshape, just like training data
X_test_feature = np.expand_dims(X_test_feature, axis=0)
test_for_RF = np.reshape(X_test_feature, (x_test.shape[0], -1))

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_RF)
#print(cm)
sns.heatmap(cm, annot=True)

#TRAINING SET
X = X_for_training
y= train_labels

from datetime import datetime
# Here we go
start_time = timer(None)
n_splits = 3
kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
#train_prediction = SVM_model.predict(X)    
# Print Accuracy, Confusion Matrix, Classification Report
for train_index, val_index in kf.split(X_for_training, y):
    classifier.fit(X_for_training[train_index], y[train_index])
    train_prediction = classifier.predict(X_for_training[val_index])    
    #train_prediction = le.inverse_transform(train_prediction)
    print ("Accuracy = ", metrics.accuracy_score(y[val_index], train_prediction))
    #print(cross_val_score(SVM_model, X_for_training, y_train, cv=5))
    print(confusion_matrix(y[val_index], train_prediction))
    print(classification_report(y[val_index], train_prediction))
timer(start_time) # timing ends here for "start_time" variable


#TEST SET
X = test_for_RF
y= test_labels

from datetime import datetime
# Here we go
start_time = timer(None)
n_splits = 3
kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
#train_prediction = SVM_model.predict(X)    
# Print Accuracy, Confusion Matrix, Classification Report
for test_index, vali_index in kf.split(test_for_RF, y):
    classifier.fit(test_for_RF[test_index], y[test_index])
    test_prediction = classifier.predict(test_for_RF[vali_index])    
    #train_prediction = le.inverse_transform(train_prediction)
    print ("Accuracy = ", metrics.accuracy_score(y[vali_index], test_prediction))
    #print(cross_val_score(SVM_model, test_for_RF, y_train, cv=5))
    print(confusion_matrix(y[vali_index], test_prediction))
    print(classification_report(y[vali_index], test_prediction))
timer(start_time) # timing ends here for "start_time" variable


#MOdel fitting
start_time = timer(None)
xgboost_model = classifier.fit(X_for_training,y_train)
timer(start_time) # timing ends here for "start_time" variable

#Now predict using the trained RF model. 
start_time = timer(None)
prediction_RF = xgboost_model.predict(test_for_RF)
#Inverse le transform to get original label back. 
prediction_RF = le.inverse_transform(prediction_RF)
timer(start_time) # timing ends here for "start_time" variable

#Check results on a few select images
start_time = timer(None)
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature= VGG_model.predict(input_img)
input_img_features = np.expand_dims(input_img_feature, axis=0)
input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
prediction_RF = xgboost_model.predict(input_img_for_RF)
prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", test_labels[n])
timer(start_time) # timing ends here for "start_time" variable























