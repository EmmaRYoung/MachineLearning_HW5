import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import img_to_array, array_to_img


#import pandas as pd
#import seaborn as sn
#import matplotlib.pyplot as plt


#implement from scratch
import keras, os
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.datasets import mnist


from tensorflow.keras.optimizers import Adam 

from keras.callbacks import ModelCheckpoint, EarlyStopping
#must reshape images so they cn be passed into vgg16

import matplotlib.pyplot as plt
import cv2 as cv


(xtrain,ytrain),(xtest,ytest)= mnist.load_data()

xtrain=np.dstack([xtrain] * 3)
xtest=np.dstack([xtest]*3)
xtrain.shape,xtest.shape

# Reshape images as per the tensor format required by tensorflow

xtrain = xtrain.reshape(-1, 28,28,3)
xtest= xtest.reshape (-1,28,28,3)
xtrain.shape,xtest.shape

# As 48 the size acceptable not 28

# Resize the images 48*48 as required by VGG16

xtrain = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in xtrain])
xtest = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in xtest])
#train_x = preprocess_input(x)
xtrain.shape, xtest.shape


# Convert the labels to one-hot vectors
y_train = to_categorical(ytrain)
y_test = to_categorical(ytest)

'''
x_train_gray = x_train.reshape(60000, 28,28,1).astype("float32") / 255

x_test_gray = x_test.reshape(10000, 28,28,1).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

#create training, testing, and validation images
training_images = np.empty([1000, 48, 48, 3])
y_train0 = y_train[0:1000]

testing_images = np.empty([1000, 48, 48, 3])
y_test0 = y_test[0:1000]


for i in range(1000):
    temp = np.concatenate((x_train_gray[i],x_train_gray[i],x_train_gray[i]),axis = 2)
    training_images[i,:,:,:] = cv.resize(temp, (48, 48), interpolation = cv.INTER_AREA)
    
    temp2 = np.concatenate((x_test_gray[i],x_test_gray[i],x_test_gray[i]),axis = 2)
    testing_images[i,:,:,:] = cv.resize(temp2, (48, 48), interpolation = cv.INTER_AREA)


y_train0 = to_categorical(y_train0)
y_test0 = to_categorical(y_test0)
'''


model = Sequential() #no weights = randomly initialize network
'''
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3), kernel_initializer = 'random_normal', padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=10, activation="softmax"))
'''

model.add(Conv2D(64, (3, 3), input_shape=(48, 48, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


opt = Adam(lr = 0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

#model.summary()
hist = model.fit(xtrain, y_train, epochs = 10)

y_predict = model.predict(xtest, y_test.all(), verbose=2)