# INF2102 - Programming Conclusion Project

**2020.1**

Discipline of the postgraduate program (M.Sc.) at the Department of Informatics of PUC-Rio.


# Installation and Execution

### Installing in Anaconda environment

We can use Anaconda to set an environment.

```bash
conda create -n <environment_name> python=3.6
conda activate <environment_name>
conda install tensorflow-gpu==2.1.0
```


### Install the dependencies of the project through the command

Then, locate the project's root directory and use pip to install the requirements (`requirements.txt`).

```bash
pip install -r requirements.txt
```

### Downloading x-ray data

- To download the dataset, it's required a Kaggle's API, which it's described here how to get one: https://github.com/Kaggle/kaggle-api. 
Once it is done, the json file should be located in _intput/_ directory, named as "kaggle.json".

or 

- It can be downloaded manually on Kaggle here: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia and extracted to a folder named "xray_data/" in the project's repository.

### Project Structure

```
├── requirements                            - Specifies the library dependencies.
|
├── raw_data                                - Contains the data to be downloaded.
|
├── input                                   - This folder contains the models' inputs.
|
├── output                                  - This folder contains the partial and final results.
|
├── data_loader                             - Package for data loader.
│   ├── xray_loader.py                      - Pneumonia x-ray dataset loader.
|
├── model                                   - Package for ConvNet model definition, training and evaluation.
│   └── conv_model.py                       - Module to construct the Convolutional Network model.
│   └── TL_model.py                         - Module to construct the model from Tranfer Learning.
|
├── preprocessing                           - Package for preconfiguration of main module.
│   └── preconfiguration.py                 - Preconfigures Tensowflow-GPU and downloads data if needed.
|
├── previsualization                        - Package for previsualizing x-ray data.
│   └── preplotter.py                       - Plots data before model training.
|
├── utils                                   - Package for utilities.
|   ├── plotter.py                          - Module for plotting visual utilities and results.
```

### Report

* [INF2102-PFP-LuisEduardoCraizer.pdf](https://github.com/lecraizer/chest_xray/blob/master/docs/INF2102-PFP-LuisEduardoCraizer.pdf)
