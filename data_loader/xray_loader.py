#!/usr/bin/env python3
"""
This class loads the dataset from the xray_data folder and process to train, val, test sets.
    Created on Mon Jun 1 2020
@author: Luis Eduardo Craizer
@version: 1.0
"""

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from skimage.transform import resize
from tqdm import tqdm
import numpy as np
import cv2    
import os               


class DataLoader():      
        
    def __init__(self):
        """
        Initialize the absolute path of each set.
        """
        self.TRAIN_DIR = "xray_data/chest_xray/train/"
        self.TEST_DIR =  "xray_data/chest_xray/test/"
        
        
    def read_files(self, DIR):
        """
        Construct the structure to extract images from the X-ray directories.
        """
        X = []
        y = []

        for nextDir in os.listdir(DIR):
            if not nextDir.startswith('.'):
                if nextDir in ['NORMAL']:
                    label = 0
                elif nextDir in ['PNEUMONIA']:
                    label = 1

                temp = DIR + nextDir
                for file in tqdm(os.listdir(temp)):
                    img = cv2.imread(temp + '/' + file)
                    if img is not None:
                        img = resize(img, (150, 150, 3)) # resize image to standard shape
                        img = np.asarray(img) # transform image to a numpy array
                        X.append(img)
                        y.append(label)

        X = np.asarray(X)
        y = np.asarray(y)
        
        # transforming y variables to categorical
        y = to_categorical(y, 2)
        return X,y
    
    
    def get_data(self):
        """
        Read images from the directories saving them into matrices.
        """
        X_train, y_train = self.read_files(self.TRAIN_DIR) # read images from TRAIN dir
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, stratify=y_train, test_size=0.1) # splits train and validation set
        
        print('X_train:', X_train.shape,'\ny_train:',y_train.shape,'\n')
        print('X_val:', X_val.shape,'\ny_val:',y_val.shape,'\n')             
        
        X_test, y_test = self.read_files(self.TEST_DIR) # read images from TEST dir
        print('X_test:', X_test.shape,'\ny_test:',y_test.shape,'\n')
        
        return X_train, y_train, X_val, y_val, X_test, y_test

