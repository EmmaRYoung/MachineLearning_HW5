import numpy as np
import pandas as pd
import seaborn as sn


import tensorflow as tf
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt

#import pandas as pd
#import seaborn as sn
#import matplotlib.pyplot as plt


#import scipy.io
#mat = scipy.io.loadmat('mnist_all.mat')

#implement from scratch
import keras, os
#from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model
#from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import optimizers 

import cv2 as cv


def ReportAccuracy(Confusion):
    #obtain useful information from the confusion matrix
    #right now reports accuracy, but can report anything as TP, TN, FN, FP are calculated first
    dim = len(Confusion)
    TPstore = np.zeros((dim,1))
    FNstore = np.zeros((dim,1))
    FPstore = np.zeros((dim,1))
    TNstore = np.zeros((dim,1))
    AccurStore = np.zeros((dim,1))

    for i in range(dim):
        TPstore[i] = Confusion[i,i]
        
        beforei_R = Confusion[i,0:i]
        afteri_R = Confusion[i,i+1:dim]
        beforei_C = Confusion[0:i,i]
        afteri_C = Confusion[i+1:dim,i]
        temp = np.delete(Confusion, obj = i, axis=0)
        AllXcept = np.delete(temp, obj = i , axis=1)
        
        FNstore[i] = np.sum(beforei_R) + np.sum(afteri_R)
        FPstore[i] = np.sum(beforei_C) + np.sum(afteri_C)
        TNstore[i] = np.sum(AllXcept)
        
        AccurStore[i] = (TPstore[i] + TNstore[i])/(TPstore[i] + TNstore[i] + FPstore[i] + FNstore[i])
        
    return AccurStore



#must reshape images so they cn be passed into vgg16

#used this github answer for help:
# https://github.com/keras-team/keras/issues/4465    

#create input
img_input = Input(shape = (28,28,3))
input2 = Input(shape = 1000)

# Generate a model with all layers (with top)
vgg16 = VGG16(weights="imagenet", include_top=True)

#Add a layer where input is the output of the  second last layer 
x = Dense(10, activation='softmax', name='predictions')(vgg16.layers[-2].output)

#Then create the corresponding model 
model = Model(vgg16.input, x)
#model.summary()

#begin training model, import data
# help from this website: https://www.tensorflow.org/guide/keras/train_and_evaluate
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_val = x_train[-1000:]
y_val = y_train[-1000:]


# Preprocess the data (these are NumPy arrays)
x_train_gray = x_train.reshape(60000, 28,28,1).astype("float32") / 255
x_validation_gray = x_val.reshape(1000, 28, 28, 1).astype("float32") / 255

x_test_gray = x_test.reshape(10000, 28,28,1).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

#display images to make sure they're in the right format
plt.figure()
# plt.imshow(train_images_gray[0])
image=np.concatenate((x_train_gray[1],x_train_gray[1],x_train_gray[1]),axis = 2)
plt.imshow(image)
plt.colorbar()
plt.grid(False)
plt.show()

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]

#y_val = y_train[-10000:]

#x_train = x_train[:-10000]
#y_train = y_train[:-10000]


#create training, testing, and validation images
training_images = np.empty([1000, 224, 224, 3])
y_train0 = y_train[0:1000]

validation_images = np.empty([1000, 224, 224, 3])
y_val0 = y_val[0:1000]

testing_images = np.empty([1000, 224, 224, 3])
y_test0 = y_test[0:1000]

for i in range(1000):
    temp = np.concatenate((x_train_gray[i],x_train_gray[i],x_train_gray[i]),axis = 2)
    training_images[i,:,:,:] = cv.resize(temp, (224, 224), interpolation = cv.INTER_AREA)
    
    temp1 = np.concatenate((x_validation_gray[i],x_validation_gray[i],x_validation_gray[i]),axis = 2)
    validation_images[i,:,:,:] = cv.resize(temp1, (224, 224), interpolation = cv.INTER_AREA)
    
    temp2 = np.concatenate((x_test_gray[i],x_test_gray[i],x_test_gray[i]),axis = 2)
    testing_images[i,:,:,:] = cv.resize(temp2, (224, 224), interpolation = cv.INTER_AREA)

plt.figure()
# plt.imshow(train_images_gray[0])
image=np.concatenate((x_train_gray[1],x_train_gray[1],x_train_gray[1]),axis = 2)
plt.imshow(image)
plt.colorbar()
plt.grid(False)
plt.show()
    

model.compile(
    optimizer=optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

model.summary()
#freeze convolutional layers
layerNum = [20, 21]
for i in range(len(layerNum)):
    n = layerNum[i]
    model.layers[n].trainable = False


print("Fit model on training data")
history = model.fit(
    training_images,
    y_train0,
    batch_size=13,
    epochs=10,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(validation_images, y_val0),
)


#predict
y_test = np.reshape(y_test0, (1000,1))
test = model.predict(testing_images, y_test.any())

#convert to a pandas dataframe
prediction_df = pd.DataFrame(test)
#find the index of the maximum in each row
prediction_vect = prediction_df.idxmax(axis=1)

#create confusion matrix
Confusion = np.zeros((10,10))
count = 0
for j in range(len(prediction_vect)):
    #true class: column
    #predicted class: row
    row_ind = prediction_vect[j]
    col_ind = np.int(y_test[j])
    if row_ind != col_ind:
        count = count+1
    
    Confusion[row_ind, col_ind] = Confusion[row_ind, col_ind] + 1


classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
dat_Confusion = pd.DataFrame(Confusion, index = classes, columns = classes)
plt.figure(figsize = (10,7))
plt.title("Confusion matrix for Keras with ImageNet Weights - Retrained on MNIST")
cfm_plot = sn.heatmap(dat_Confusion, annot=True)

Report = ReportAccuracy(Confusion)