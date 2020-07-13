#!/usr/bin/env python3
"""
Preconfigures the environment for the project to run.
    Created on Wed Jun 3 2020
@author: Luis Eduardo Craizer
@version: 1.0
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


class PreProcessor():    
    
    def set_gpu_limit(self):
        """ 
        Set limit to each GPU in jupyter lab GPU processses.
        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)


    def download_files(self):
        """
        Download kaggle x-ray dataset and unzip it if not done yet.
        """
        if not os.path.isfile('chest-xray-pneumonia.zip'):
            # case dataset zipfile haven't been downloaded yet
            os.system('pip install -q kaggle')
            os.system('mkdir -p ~/.kaggle')
            os.system('cp input/kaggle.json ~/.kaggle/') # necessary JSON file to download a kaggle dataset
            os.system('kaggle datasets download -d paultimothymooney/chest-xray-pneumonia/')
            
        if not os.path.isdir('xray_data/'):
            # case dataset haven't been unzipped yet
            os.system('unzip chest-xray-pneumonia.zip')
            os.rename('chest_xray', 'xray_data')

            
            
