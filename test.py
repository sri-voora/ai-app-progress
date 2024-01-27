import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam

import numpy as np
import pandas as pd

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib
from matplotlib import pyplot as plt
#matplotlib inline
#config InlineBackend.figure_format = 'retina'

import os
import glob
import random
from skimage.io import imread

directory_path = '/Users/sri.voora/Desktop/gfg'

tumor_images = glob.glob(os.path.join(directory_path, '**', '*.png'), recursive=True)

class0 = []
class1 = []

for filename in tumor_images:
    base_filename = os.path.basename(filename)
    
    if "class0" in base_filename:
        class0.append(filename)
    elif "class1" in base_filename:
        class1.append(filename)

print("Class 0:", 3)
print("Class 1:", 3)

from random import choice
fig,axes = plt.subplots(figsize=(20,25))

random_image = choice(class0)

fig.delaxes(axes)
fig.suptitle('Random Samples',fontsize=30)

from random import choice 
fig,axes = plt.subplots(figsize=(20,25))
  
"""random_image = choice(class1)


for i in range(0,len(random_image)):
    img = imread(random_image[i])
    plt.imshow(img)
    axes =plt.subplot(5,5,i+1)
    axes.set_title('patientID:'+str(random_image[i][-5]),fontsize=20)


batch_size = 32

img_height = 180 
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  './16896',
  validation_split=0.2, 
  subset="training",
  seed=123, #random seed number
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  './16896',
  validation_split=0.2,
  subset="validation",
  seed=123, #random seed number
  image_size=(img_height, img_width),
  batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

num_classes = 2 

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])"""