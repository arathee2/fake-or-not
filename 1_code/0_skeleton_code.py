# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:53:46 2020

@author: jdavi
"""
# Basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# Image processing and ELA conversion
from PIL import Image
import os
from pylab import *
import re
from PIL import Image, ImageChops, ImageEnhance

# General ML models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from pandas_ml import ConfusionMatrix


# CNN using my laptop GPU
import tensorflow as tf
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

sns.set(style='white', context='paper', palette='deep')
np.random.seed(2)



def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]


def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    ELA_filename = filename.split('.')[0] + '.ela.png'

    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)

    ela_im = ImageChops.difference(im, resaved_im)

    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    return ela_im

paths = get_imlist('D:/ML final project/PS_Battles/dataset/')

dataset = pd.DataFrame(paths, columns = ['file_name'])

dataset['fake'] = 0

dataset['fake'] = np.where(dataset['file_name'].str.endswith('_fake.jpg'), 1, dataset['fake'])

pd.crosstab(index = dataset['fake'], columns = 'Freq')/len(paths)*100
# Load pre-processed images

X = []
Y = []

new_index = []
for index, row in dataset.iterrows():
    try:
        X.append(array(convert_to_ela_image(row[0], 90).resize((128, 128))).flatten() / 255.0)
        Y.append(row[1])
        new_index.append(index)
    except:
        print("Image {} dropped. Could not be processed.".format(re.sub('D:/ML final project/PS_Battles/dataset/', '', dataset['file_name'][index])))
        pass

    if index % 1000 == 0:

        print(index, "images parsed")

dataset_correct = dataset[dataset.index.isin(new_index)]
dataset_correct = dataset_correct.reset_index(drop=True)


X = np.array(X)
X = X.reshape(-1, 128, 128, 3)

Y = tf.keras.utils.to_categorical(Y, 2)


# np.save("D:/ML final project/PS_Battles/ps_battles_ela_X.npy", X)

# np.save("D:/ML final project/PS_Battles/ps_battles_ela_Y.npy", Y)

# dataset_correct.to_csv('D:/ML final project/PS_Battles/dataset_correct.csv')


'''
*******************************************************************************
*******************************************************************************
********************  LOAD .npy FILES AT THIS POINT  **************************
*******************************************************************************
*******************************************************************************
'''


'''
X = np.load("D:/ML final project/PS_Battles/ps_battles_ela_X.npy")
Y = np.load("D:/ML final project/PS_Battles/ps_battles_ela_Y.npy")
dataset_correct = pd.read_csv("D:/ML final project/PS_Battles/dataset_correct.csv")
'''

def plot_image(img_num):

    file_name = dataset_correct['file_name'][img_num]
    img_name = re.sub('D:/ML final project/PS_Battles/dataset/', '', file_name)

    plt.imshow(X[img_num], interpolation='nearest')
    plt.title("ELA version of image {}".format(img_name))
    plt.show()
    return img_name

# Cheerleaders fake image
im_name = plot_image(2220)

Image.open('D:/ML final project/PS_Battles/dataset/'+im_name)




X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)


# First CNN architecture building:



# Release GPU memory

tf.keras.backend.clear_session()


try:
    del model
except:
    print("No model in memory")





model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid',
                     activation ='relu', input_shape = (128,128,3)),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid',
                     activation ='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = "relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation = "softmax")
])


model.summary()

optimizer = tf.keras.optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])




# I test two stopping criterias:

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()




epochs = 30
batch_size = 100


history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,
          validation_data = (X_val, Y_val), callbacks=[callbacks])







# Plot the loss and accuracy curves for training and validation

fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_title("Evolution of loss by Epoch")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax[1].set_ylabel("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_title("Evolution of accuracy by Epoch")
legend = ax[1].legend(loc='best', shadow=True)




# Confussion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred,axis = 1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2))

# ~70% Accuracy
