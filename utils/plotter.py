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
        self.DIR = "xray_data/chest_xray/train/"
        self.PNEU_DIR = os.listdir(self.DIR + "PNEUMONIA")
        self.REG_DIR = os.listdir(self.DIR + "NORMAL")

        
    def image_visualizer(self, n):
        """
        Plot n pairs of regular and pneumonia x-ray images.
        """
        for i in range(0,5):
            imagep1 = cv2.imread(self.DIR+"PNEUMONIA/"+self.PNEU_DIR[i])
            imagep1 = skimage.transform.resize(imagep1, (150, 150, 3), mode = 'reflect')
            imagen1 = cv2.imread(self.DIR+"NORMAL/"+self.REG_DIR[i])
            imagen1 = skimage.transform.resize(imagen1, (150, 150, 3))
            pair = np.concatenate((imagen1, imagep1), axis=1)
            plt.figure(figsize=(10,5))
            plt.imshow(pair)
            plt.imsave("output/images_output_" + str(i) + ".png", pair)

            
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
        plt.savefig('output/history_results.png')
        
        
    def plot_confusion_matrix(self, y_true, pred):
        """
        Generate a confusion matrix plot.
        """
        plt.clf() # erase plot
        conf_matrix = confusion_matrix(y_true, pred)
        fig, ax = plot_confusion_matrix(conf_mat=conf_matrix ,  figsize=(5, 5))
        fig.savefig('output/confusion_matrix.png')
        
        
    def plot_countplot(self, data):
        """
        Show the difference in regular and pneumonia exams quantities in a countplot.
        """
        plt.clf() # erase plot
        count = data.sum(axis = 0)
        sns_plot = sns.countplot(x = count)
        sns_plot.get_figure().savefig("output/countplot.png")