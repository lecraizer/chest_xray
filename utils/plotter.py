#!/usr/bin/env python3
"""
Generates plots containing results of the experiment.
    Created on Wed Jun 10 2020
@author: Luis Eduardo Craizer
@version: 1.0
"""

import matplotlib 
matplotlib.use('Agg') 
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import skimage
import cv2
import os


class Visualizer():
    
    def __init__(self):
        """
        Initialize the directories.
        """
        self.OUTPUT_DIR = "output/"
            
    def history_results(self, hist):
        """
        Plot the results of the model training.
        """
        plt.clf() # erase plot
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        # Summarize history for loss
        plt.clf() # erase plot
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.OUTPUT_DIR + 'history_results.png')
        
        
    def plot_confusion_matrix(self, y_true, pred):
        """
        Generate a confusion matrix plot.
        """
        plt.clf() # erase plot
        conf_matrix = confusion_matrix(y_true, pred)
        fig, ax = plot_confusion_matrix(conf_mat=conf_matrix ,  figsize=(5, 5))
        fig.savefig(self.OUTPUT_DIR + 'confusion_matrix.png')
        
        
