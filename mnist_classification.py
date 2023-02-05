# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 19:43:50 2023

@author: dlinogarda.com
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# plt.imshow(x_train[0,:,:])

# Jumlah label 
num_labels = len(np.unique(y_train))

# dikonvert ke one-hot vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# dimensi gambar(diasumsikan square)
image_size = x_train.shape[1]
input_size = image_size * image_size

# resize dan normalisasi
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255

x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

# mengubah dimensi image ke dalam bentuk semula
x_test_img = np.reshape(x_test, [-1, image_size,image_size])

# menentukan parameter network
batch_size = 256
hidden_units = 256
dropout = 0.45

# model dengan 3 layer (2 layer adalah hidden layer, 1 output) + ReLU, Dropout
model = Sequential()
model.add(Dense(hidden_units, input_dim = input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels))
# output untuk one-hot vector
model.add(Activation('softmax'))
model.summary()
plot_model(model, to_file='network_architecture.png', show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(x_test,y_test,epochs = 10, batch_size=batch_size)

Result = model.predict(x_train)
plt.imshow(x_test_img[999])
print(np.argmax(Result[999]))












