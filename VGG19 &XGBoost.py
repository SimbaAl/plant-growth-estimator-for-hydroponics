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
from mpl_toolkits.mplot3d import Axes3D

print(os.listdir('canola_training'))

#Resize images to
SIZE = 224


#creating empty lists to input train images and train labels.
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



# Creating empty lists to input test images and labels.
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

#relabelling the test and train datasets
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

###################################################################
# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0


#Load model wothout classifier/fully connected layers
VGG_model = VGG19(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0

#Extraction of features convolutional network begins
from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

#Feature extraction
start_time = timer(None) # timing starts from this point for "start_time" variable
feature_extractor=VGG_model.predict(x_train)
timer(start_time) 
features = feature_extractor.reshape(feature_extractor.shape[0], -1) #reshaping feature data

X_for_training = features #This is our X input to XGBoost

#Classifier and parameters defined
classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7,
              enable_categorical=False, gamma=0.6, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.5, max_delta_step=0, max_depth=8,
              min_child_weight=7, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)


#TRAINING using cross validation starts
X = X_for_training
y= train_labels

start_time = timer(None) #Timing starts
n_splits = 3 #3 fold cross validation
kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    
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

#MOdel fitting
start_time = timer(None)
xgboost_model = classifier.fit(X_for_training,y_train)
timer(start_time) # timing ends here for "start_time" variable

#Send test data through same feature extractor process
start_time = timer(None)
X_test_feature = VGG_model.predict(x_test)
timer(start_time) # timing ends 
X_test_feature = np.expand_dims(X_test_feature, axis=0)
test_for_XG = np.reshape(X_test_feature, (x_test.shape[0], -1)) #reshaping the test features



#Prediction on TEST SET
X = test_for_XG
y= test_labels

start_time = timer(None) #testing starts
n_splits = 3
kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
#train_prediction = SVM_model.predict(X)    
# Print Accuracy, Confusion Matrix, Classification Report
for test_index, vali_index in kf.split(test_for_XG, y):
    classifier.fit(test_for_XG[test_index], y[test_index])
    test_prediction = classifier.predict(test_for_XG[vali_index])    
    #train_prediction = le.inverse_transform(train_prediction)
    print ("Accuracy = ", metrics.accuracy_score(y[vali_index], test_prediction))
    #print(cross_val_score(SVM_model, test_for_XG, y_train, cv=5))
    print(confusion_matrix(y[vali_index], test_prediction))
    print(classification_report(y[vali_index], test_prediction))
timer(start_time) # timing ends


#Now predict using the trained XGBoost model. 
start_time = timer(None)
prediction_XG = xgboost_model.predict(test_for_XG)
#Inverse le transform to get original label back. 
#prediction_XG = le.inverse_transform(prediction_XG)
timer(start_time) # timing ends here for "start_time" variable

#Predict on randomly select images
start_time = timer(None)
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expanding dimensions so the input is (num images, x, y, c)
input_img_feature= VGG_model.predict(input_img)
input_img_features = np.expand_dims(input_img_feature, axis=0)
input_img_for_XG = np.reshape(input_img_features, (input_img.shape[0], -1))
prediction_XG = xgboost_model.predict(input_img_for_XG)
#prediction_XG = le.inverse_transform([prediction_XG])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_XG)
print("The actual label for this image is: ", test_labels[n])
timer(start_time) # timing ends here for "start_time" variable

##set up the parameters
rcParams['figure.figsize'] = 150, 100
plot_tree(classifier,num_trees=2)






















