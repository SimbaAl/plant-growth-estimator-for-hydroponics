# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:20:03 2021

@author: 36450057
"""

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
from tensorflow.keras.applications import VGG16
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
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print(os.listdir('canola_dataset'))

#Resize images to
SIZE = 128


#creating empty lists.
train_orig_images = []
train_orig_labels = [] 

for directory_path in glob.glob('canola_dataset/work/*'):
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


# masked dataset


train_mask_images = []
train_mask_labels = [] 

for directory_path in glob.glob('canola_dataset/work/*'):
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

        
    

dataset_images = np.concatenate([train_mask_images, train_orig_images], axis=1)    #Concatenate both image and mask datasets
#dataset_labels=np.concatenate([train_mask_labels, train_orig_labels], axis=1) 
#Encode labels from text (folder names) to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
#le.fit(train_mask_labels)
#test_labels_encoded = le.transform(train_mask_labels)
le.fit(train_orig_labels)
train_labels_encoded = le.transform(train_orig_labels)

x_train, y_train = train_orig_images, train_labels_encoded

# Normalize pixel values to between 0 and 1
x_train = x_train / 255.0




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


#Extract features from training images

image_features = feature_extractor(x_train)

n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
X_for_RF = np.reshape(image_features, (x_train.shape[0], -1))  #Reshape to #images, features

np.isnan(X_for_RF)
np.where(np.isnan(X_for_RF))

df = X_for_RF


X=df
y=train_orig_labels


feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])


#PCA
pca = PCA(n_components=5)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]
#df['pca-four'] = pca_result[:,3]
#df['pca-stage_4'] = pca_result[:,4]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 5),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

#3D PCA visualisation 
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["pca-one"], 
    ys=df.loc[rndperm,:]["pca-two"], 
    zs=df.loc[rndperm,:]["pca-three"], 
    c=df.loc[rndperm,:]["y"], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()

#TSNE  
N = 3500
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-background'] = pca_result[:,0]
df_subset['pca-stage_1'] = pca_result[:,1] 
df_subset['pca-stage_2'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 5),
    data=df_subset,
    legend="full",
    alpha=0.3
)

plt.figure(figsize=(16,7))
ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x="pca-background", y="pca-stage_1",
    hue="y",
    palette=sns.color_palette("hls", 5),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 5),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(data_subset)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]
plt.figure(figsize=(7,16))
ax1 = plt.subplot(3, 1, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 5),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(3, 1, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 5),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
ax3 = plt.subplot(3, 1, 3)
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette("hls", 5),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax3
)

