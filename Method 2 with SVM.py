# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:16:29 2021

@author: 36450057
"""
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import xgboost
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import cv2
import os
import random
import sklearn
import pandas as pd
from skimage import io, filters, feature
from skimage.filters import sobel
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC

print(os.listdir('canola_training'))

#Resizing images to
SIZE = 128

#Training set lists.
train_orig_images = []
train_orig_labels = [] 

for directory_path in glob.glob('canola_training/training/*'):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.bmp")):
        print(img_path)
        orig_img   = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        orig_img   =cv2.resize(orig_img, (SIZE, SIZE)) 
        
        train_orig_images.append(orig_img)
        train_orig_labels.append(label)
        
train_orig_images = np.array(train_orig_images)
train_orig_labels = np.array(train_orig_labels)  


# train masked dataset list
train_mask_images = []
train_mask_labels = [] 

for directory_path in glob.glob('canola_training/training/*'):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.bmp")):
        print(img_path)
        img   = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        img   =cv2.resize(img, (SIZE, SIZE)) 
        
        
        elem_open  = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) # create a 5x5 processing filter
                                                                         # for opening 
        elem_close = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        img_open   = cv2.morphologyEx(img,cv2.MORPH_OPEN ,elem_open ) # Image with open morphological op
 
        img_filt        = cv2.morphologyEx(img_open,cv2.MORPH_CLOSE,elem_close) # Closing morphological op
        # Threshold to binary
        ret, im_bin     = cv2.threshold(img_filt,40,1,cv2.THRESH_BINARY)
    
        my_contours, hierarchy = cv2.findContours(im_bin.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        mask_ext = cv2.drawContours(im_bin.copy(),my_contours,-1,255) #,self.thickness)
        mask = cv2.drawContours(mask_ext.copy(),my_contours,-1,100,2)
        my_masked_orig = np.where((mask_ext==255),img,0)
        train_mask_images.append(my_masked_orig)
        train_mask_labels.append(label)

train_mask_images = np.array(train_mask_images)
train_mask_labels = np.array(train_mask_labels) 

# test dataset list
test_images = []
test_labels = [] 

for directory_path in glob.glob('canola_test/test/*'):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.bmp")):
        print(img_path)
        orig_img   = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        orig_img   =cv2.resize(orig_img, (SIZE, SIZE)) 
        #orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        test_images.append(orig_img)
        test_labels.append(label)
        
test_images = np.array(test_images)
test_labels  = np.array(test_labels )          
    
#CONCATENATING THE MASKS AND THE ORIGINAL IMAGES
dataset_images = np.concatenate([train_mask_images, train_orig_images], axis=1)    #Concatenate both image and mask datasets

#label encorder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(train_mask_labels)
train_mask_labels_encoded = le.transform(train_mask_labels)
le.fit(train_orig_labels)
train_labels_encoded = le.transform(train_orig_labels)
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)

#Assigning the dataset into meaningful convention
x_train, y_train, x_test, y_test = dataset_images, train_labels_encoded, test_images, test_labels_encoded


###################################################################
# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0


#Feature extraction Stage
def feature_extractor(dataset):
    x_train = dataset
    image_dataset = pd.DataFrame()
    for image in range(x_train.shape[0]):  #iterate through each image         
        
        df = pd.DataFrame()  #Store the feature information itno a temporary data frame 
             
        input_img = x_train[image, :,:]
        img = input_img
        pixel_values = img.reshape(-1)
        df['Pixel_Value'] = pixel_values   #The first feature is the pixel value
        
        # FEATURE 2 - Gabor filter responses
        #Generating Gabor kernels
        num = 1  #Starting a counter to give Gabor features a lable in the data frame
        kernels = []      
        for theta in range(4):   #Defining the number of thetas
            theta = theta / 4. * np.pi
            for lamda in np.arange(np.pi/10, np.pi / 8):   #Range of wavelengths
                gamma = 0.5
                sigma = 0.5
                #phi = 0  
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                #filtering the image and adding values to a new column 
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  #Labeling columns as Gabor1, Gabor2,Gabor3, and Gabor4
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
        
        # FEATURE 3 Sobel
        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df['Sobel'] = edge_sobel1
                       
        image_dataset = image_dataset.append(df)
        
    return image_dataset

#Starting and timing the Feature extraction using the Feature extractor defined in the feature extraction stage
from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


#Extracting features from training images
start_time = timer(None)
image_features = feature_extractor(x_train)
timer(start_time) # timing ends here for "start_time" variable    
n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
X_for_SVM = np.reshape(image_features, (x_train.shape[0], -1))  #Reshape to #images, features

#Cheching for NaNs
np.isnan(X_for_SVM)
np.where(np.isnan(X_for_SVM))

#CLASSIFIER
SVM_model= SVC(kernel='rbf', random_state=1, gamma=0.00001, C=32)
#SVM_model= SVC(kernel='poly', random_state=0, gamma=0.00001, C=30, degree=2)


#Training time and CROSS VALIDATION ON TRAINING DATASET X_for_SVM 
X = X_for_SVM
y= train_orig_labels

# Here we go
start_time = timer(None)
n_splits = 3
kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

for train_index, val_index in kf.split(X_for_SVM,y):
    SVM_model.fit(X_for_SVM[train_index], y[train_index])
    train_prediction = SVM_model.predict(X_for_SVM[val_index])    
    #train_prediction = le.inverse_transform(train_prediction)
    print ("Accuracy = ", metrics.accuracy_score(y[val_index], train_prediction)) 
    print(confusion_matrix(y[val_index], train_prediction))
    print(classification_report(y[val_index], train_prediction))
timer(start_time) # timing ends here for "start_time" variable    


#Extract features from test data and reshape, just like training data
start_time = timer(None)
X_test_feature = feature_extractor(x_test)
X_test_feature = np.expand_dims(X_test_feature, axis=0)
test_for_SVM = np.reshape(X_test_feature, (x_test.shape[0], -1))
timer(start_time) # timing ends here for "start_time" variable

# Capturing the fitting time
start_time = timer(None) # timing starts
SVM_model.fit(X_for_SVM, train_orig_labels)
timer(start_time) # timing ends 


#TEST SET
X = test_for_SVM
y= test_labels

#Testing time
start_time = timer(None)
n_splits = 3
kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
  
# Print Accuracy, Confusion Matrix, Classification Report
for test_index, vali_index in kf.split(test_for_SVM, y):   
    SVM_model.fit(test_for_SVM[test_index], y[test_index])
    test_prediction = SVM_model.predict(test_for_SVM[vali_index])    
    #train_prediction = le.inverse_transform(train_prediction)
    print ("Accuracy = ", metrics.accuracy_score(y[vali_index], test_prediction))
    print(confusion_matrix(y[vali_index], test_prediction))
    print(classification_report(y[vali_index], test_prediction))
timer(start_time) # timing ends here for "start_time" variable


#Now predict using the trained SVM model. 
start_time = timer(None)
prediction_SVM = SVM_model.predict(test_for_SVM)
#Inverse le transform to get original label back. 

timer(start_time) # timing ends here for "start_time" variable

#Now predict using the trained SVM model. 
prediction_SVM = SVM_model.predict(test_for_SVM)
#Inverse le transform to get original label back. 
#prediction_SVM = le.inverse_transform(prediction_SVM)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_SVM))


#Check results on a few select images
start_time = timer(None)
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature= feature_extractor(input_img)
input_img_features = np.expand_dims(input_img_feature, axis=0)
input_img_for_SVM = np.reshape(input_img_features, (input_img.shape[0], -1))
prediction_SVM = SVM_model.predict(input_img_for_SVM)
#prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_SVM)
print("The actual label for this image is: ", test_labels[n])
timer(start_time) # timing ends here for "start_time" variable
