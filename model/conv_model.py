#!/usr/bin/env python3
"""
ConvNet model for chest x-ray dataset.
    Created on Mon Jun 8 2020
@author: Luis Eduardo Craizer
@version: 1.0
"""

from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau 
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras import backend as K
import numpy as np
import time


class ConvolutionalModel():
    
    def __init__(self):
        """
        Initialize model parameters and hiperparameters.
        """
        self.HEIGHT = 150 # input height 
        self.WIDTH = 150 # input width
        self.DEPTH = 3 # input depth
        self.EPOCHS = 20 # number of epochs
        self.LOSS_FUNC = 'binary_crossentropy' # loss function
        self.OPT_FUNC = RMSprop(lr=0.0005) # optimization function
        self.BS = 256 # batch size
        self.METRICS = ['accuracy'] # metrics to be evaluated by the model
        self.LR_REDUCE = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=0.0001, patience=3, verbose=1) # reduce learning rate timely
        self.CHECKPOINT = ModelCheckpoint("output/weights.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min') # checkpoints the best result
        self.HIPERPARAMS = [self.LR_REDUCE, self.CHECKPOINT]
    
    
    def swish_activation(self, x):
        """
        Activation function using the sigmoid function.
        """
        return (K.sigmoid(x) * x)
    
    
    def initialize_model(self):
        """
        Initialize the construction of a sequential model.
        """
        self.model = Sequential()
        return
    
    
    def create_model(self):
        """
        Create the model layers and features.
        """
        # 16
        self.model.add(Conv2D(16, (3, 3), padding='same', input_shape=(self.HEIGHT, self.WIDTH, self.DEPTH)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(16, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # 32
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(self.HEIGHT, self.WIDTH, self.DEPTH)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # 64
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # 96
        self.model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(96, (3, 3), padding='valid'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # 128
        self.model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(128, (3, 3), padding='valid'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())

        self.model.add(Dense(64, activation=self.swish_activation))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(2 , activation='sigmoid'))
        return

    
    def compile_model(self):
        """
        Compile the model by using the defined hiperparameters.
        """
        self.model.compile(loss=self.LOSS_FUNC,
                          optimizer=self.OPT_FUNC,
                          metrics=self.METRICS)
        return self.model
    
    
    def fit_model(self, X_train, y_train, test_set):
        """
        Fit the model with the validation set.
        """
        start_time = time.time()
        result = self.model.fit(X_train, y_train, validation_data = test_set, callbacks=self.HIPERPARAMS, epochs=self.EPOCHS)
        end_time = time.time()
        self.train_time = end_time - start_time
        print( "The model took %0.3f seconds to train.\n"%self.train_time )
        return result
    
    
    def evaluate_model(self, X_test, y_test):
        """
        Predict the labels in the test set.
        """
        pred = self.model.predict(X_test)
        pred = np.argmax(pred,axis = 1) 
        y_true = np.argmax(y_test,axis = 1)
        return pred, y_true
    
    
    def train_model(self, X_train, y_train, validation_set):
        """
        Initialize, create, compile and train the model.
        """
        model = self.initialize_model()
        model = self.create_model()
        model = self.compile_model()
        return self.fit_model(X_train, y_train, validation_set)
        