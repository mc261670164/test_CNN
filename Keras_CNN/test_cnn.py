# -*-coding: utf-8 -*-
'''
Function: test CNN on Keras with the dataset of MNIST
Date:     2018.6.20
Author:   Eric.M
Email:    master2017@163.com
'''

import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

#全局变量
batch_size = 128
nb_classes = 10
epoches    = 12

#input image dimensions
img_rows, img_cols = 28, 28

#numbers of Convolutional filters will be used
nb_filters = 32

#size of pooling area for max pooling
pool_size = (2, 2)

#Convolutional kernel size
kernel_size = (3, 3)

#the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#根据不同的backend定义不同的数据格式
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test  = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test  = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_train = X_train / 255
X_test  = X_test /255
print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#转换为one-hot模型
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test  = np_utils.to_categorical(y_test, nb_classes)

#构建模型
model = Sequential()

#Convolutional layer 1
model.add(Convolution2D(nb_filters, kernel_size, padding='same', input_shape=input_shape))
#Activation layer
model.add(Activation('relu'))
#Convolutional layer 2
model.add(Convolution2D(nb_filters, kernel_size))
#Activation layer
model.add(Activation('relu'))
#pooling layer
model.add(MaxPooling2D(pool_size=pool_size))
#神经元随机失活
model.add(Dropout(0.20))
#拉成1维数据
model.add(Flatten())
#full connection layer 1
model.add(Dense(128))
#Activation layer
model.add(Activation('relu'))
#神经元随机失活
model.add(Dropout(0.5))
#full connection layer 2
model.add(Dense(nb_classes))
#SoftMax evaluation
model.add(Activation('softmax'))

#compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

#fit the model
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoches, verbose=1, validation_data=(X_test, Y_test))

#evaluate the model
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save('test_MNIST.h5')


