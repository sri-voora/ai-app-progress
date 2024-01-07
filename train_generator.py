import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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
from glob import glob
import random
from skimage.io import imread

tumorImages = glob('./16896/**/*.png', recursive = True)
class0 = []
class1 = []

# import data
for filename in tumorImages:
    if filename.endswith("class0.png"):
        class0.append(filename)
    else:
        class1.append(filename)

print("Class 0:", len(class0))
print("Class 1:", len(class1))

from random import choice
fig,axes = plt.subplots(figsize=(20,25))

random_image = [choice(class0)for x in range(0,11)]

fig.delaxes(axes)
fig.suptitle('Random Samples',fontsize=30)