from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np

class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # conv1 -- pool1
        model.add(Conv2D(20, kernel_size=5,padding='same', input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # conv2 -- pool2
        model.add(Conv2D(50, kernel_size=5,padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # FC1
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        # softmax
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model
    
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
IMG_ROWS, IMG_COLS = 28, 28
NB_CLASSES = 10
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)
    
(x_train, y_train), (x_test, y_test) = mnist.load_data()
k.set_image_dim_ordering('th')
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_train /= 255
y_train /= 255

model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, 
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print('Test accuracy:' , score[1])
