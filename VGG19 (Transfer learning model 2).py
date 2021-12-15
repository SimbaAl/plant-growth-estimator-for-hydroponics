# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 15:51:34 2021

@author: 36450057
"""

import numpy as np
import pandas as pd
import cv2
import os
import time
import pickle
from tensorflow.keras import backend as K
K.clear_session()
import itertools
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm
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
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import VGG19
from sklearn.metrics import classification_report

#get the training images
data_path = 'radish_training/training'

data_dir_list = os.listdir(data_path)

#arrange the dataset as it appears in the the folder
pip install natsort

from natsort import natsorted, ns
data_dir_list = natsorted(data_dir_list, alg=ns.PATH | ns.IGNORECASE)

img_data_list=[]


for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

    
    for img in img_list:
      input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
      input_img=cv2.resize(input_img,(224,224))
      img_data_list.append(input_img)
      
       
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

num_classes = 5

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

#allocate the labels to the respective images
labels[0:718]=0 #718
labels[718:1422]=1 #704
labels[1422:2125]=2 #702
labels[2125:2802]=3 #678
labels[2802:4094]=4 #1292


# labels =to_categorical(labels, num_classes)
labels

kf = StratifiedKFold(n_splits=3,shuffle=True)
kf.get_n_splits(img_data, labels)
print(kf)

#VGG19 model
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

output = vgg.layers[-1].output
output = layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

# freeze pre-trained model area's layer
for layer in vgg_model.layers:
    layer.trainable = False

input_shape = vgg_model.output_shape[1]

vgg_model.summary()

#timer function 
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

from datetime import datetime

# Construction the whole model
def cnn_model(x_train,x_test,y_test,y_train):
    model = Sequential()
    model.add(vgg_model)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    #model.add(Dropout(0.05))
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.025))
    model.add(Dense(5, activation='softmax'))

#categorical crossentropy optimizer 
    start_time = timer(None)
    model.compile(Adam(learning_rate=.001), loss='categorical_crossentropy', metrics=['accuracy']) 


    history=model.fit(x_train, y_train,epochs=10, batch_size=30,verbose=1)
    model.compile(loss = "categorical_crossentropy",optimizer = Adam(), metrics=['accuracy'],)
    timer(start_time)
    loss,acc = model.evaluate(x_test, y_test)
    y_prediction = model.predict(x_test)
    classes_y = np.argmax(y_prediction,axis=1)
    y_test_original=np.argmax(y_test,axis=1)
    confusion=confusion_matrix(y_true=y_test_original, y_pred=classes_y)
    target_names = ['background', 'stage_1', 'stage_2','stage_3','stage_4']
    print(classification_report(y_test_original, classes_y, target_names=target_names))
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.title('model history')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss'], loc='upper left')
    plt.show()
    plt.plot(history.history['accuracy'])
    plt.title('model history')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy'], loc='upper left')
    plt.show()
    return acc,confusion

prediction=[]
confusion_list=[]

IMAGE_WIDTH=224
IMAGE_HEIGHT=224
num_classes = 5

#print the metrics on test dataset
for train_index, test_index in kf.split(img_data, labels):
#     print("TRAIN:", train_index, "TEST:", test_index)
    start_time = timer(None)
    x_train, x_test = img_data[train_index], img_data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    x_train=x_train.reshape(x_train.shape[0],IMAGE_WIDTH,IMAGE_HEIGHT,3)
    x_test=x_test.reshape(x_test.shape[0],IMAGE_WIDTH,IMAGE_HEIGHT,3)
    y_train =to_categorical(y_train, num_classes)
    y_test =to_categorical(y_test, num_classes)
    X,Y=cnn_model(x_train,x_test,y_test,y_train)
    prediction.append(X)
    confusion_list.append(Y)
    timer(start_time)
    print(X)
    print(Y)
    #print(classification_report(y_test_original, classes_y, target_names=target_names))
    print("----------------------------")

print(prediction)

for x in range(len(confusion_list)): 
    print (confusion_list[x],sep = "\n")
    print("----------------------------")

#get the test images
test_data_path = 'radish_test/test'
test_data_dir_list = os.listdir(data_path)

from natsort import natsorted, ns
test_data_dir_list = natsorted(test_data_dir_list, alg=ns.PATH | ns.IGNORECASE)

test_img_data_list=[]


for dataset in test_data_dir_list:
    img_list=os.listdir(test_data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

    
    for img in img_list:
      input_img=cv2.imread(test_data_path + '/'+ dataset + '/'+ img )
      input_img=cv2.resize(input_img,(224,224))
      test_img_data_list.append(input_img)
      
#get the image labels      
img_data = np.array(test_img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

num_classes = 5

num_of_samples = img_data.shape[0]
label = np.ones((num_of_samples,),dtype='int64')

label[0:81]=0 #81
label[81:159]=1 #78
label[159:237]=2 #78
label[237:381]=3 #144
label[381:525]=4 #144


# labels =to_categorical(labels, num_classes)
label

prediction=[]
confusion_list=[]

IMAGE_WIDTH=224
IMAGE_HEIGHT=224
num_classes = 5

#print the metrics on the test dataset
for train_index, test_index in kf.split(test_img_data_list, label):
#     print("TRAIN:", train_index, "TEST:", test_index)
    start_time = timer(None)
    x_train, x_test = img_data[train_index], img_data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    x_train=x_train.reshape(x_train.shape[0],IMAGE_WIDTH,IMAGE_HEIGHT,3)
    x_test=x_test.reshape(x_test.shape[0],IMAGE_WIDTH,IMAGE_HEIGHT,3)
    y_train =to_categorical(y_train, num_classes)
    y_test =to_categorical(y_test, num_classes)
    X,Y=cnn_model(x_train,x_test,y_test,y_train)
    prediction.append(X)
    confusion_list.append(Y)
    timer(start_time)
    print(X)
    print(Y)
    #print(classification_report(y_test_original, classes_y, target_names=target_names))
    print("----------------------------")

print(prediction)

for x in range(len(confusion_list)): 
    print (confusion_list[x],sep = "\n")
    print("----------------------------")






































