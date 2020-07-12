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


class PreVisualizer():
    
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

            
    def plot_countplot(self, data):
        """
        Show the difference in regular and pneumonia exams quantities in a countplot.
        """
        plt.clf() # erase plot
        count = data.sum(axis = 0)
        sns_plot = sns.countplot(x = count)
        sns_plot.get_figure().savefig("output/countplot.png")