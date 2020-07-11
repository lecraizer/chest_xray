# ConvNet Implementation: An Object Oriented Approach using Keras API.

We aim to construct an object-oriented project template for working with Convolution Neural Networks (ConvNet) through Keras API. This template provides basic structures that helps to simplify the development process for new projects that aims to construct ConvNet models.

# Table Of Contents

- [About](#about): Project details.
- [Installation](#installation-and-execution): Project setup guide.
- [Architecture](#architecture): Architecture description.
    - [Project Structure](##project-structure): File arrangments in different folders.
    - [Architectural Modules](##architectural-modules): Description of the modules.
- [Implementation Example](#implementation-example): Example implementation using the template.
- [Implementation Guidelines](#implementation-guidelines): Guidelines to follow for specific Keras project implementation.
- [Further Reading](#further-reading): References for future reading.
- [FAQs](#Frequently-Asked-Questions): Most frequently asked questions. 
- [Future Works](#Future-Works): Activities in progress.
- [Acknowledgments](#acknowledgments): Acknowledgments.
- [Contributors](#contributors): Project contributors.

# About

Repository containing a deep learning supervised approach to a x-ray classification model

This project aims to construct an *Object-oriented* python code to implement Convolution Neural Networks (ConvNet) using Keras API.

A *ConvNet* consists of an input and an output layer, as well as multiple hidden layers. These layers of a ConvNet typically consist of convolutional, pooling, batch normalization, fully connected and normalization layers. 

We provide a simple keras project template that incorporates fundamentals of object-oriented programming to simplify the development process. Using this template one can easily start a ConvNet implementation without having to worry much about the helper classes and concentrate only on the construction, training and testing phases of the deep learning model.

# Installation and Execution

## Installing in Anaconda environment

We can use Anaconda to set an environment.

```bash
conda create -n <environment_name> python=3.6
conda activate <environment_name>
conda install tensorflow-gpu==2.1.0
```


## Install the dependencies of the project through the command

Then, locate the project's root directory and use pip to install the requirements (`requirements.txt`).

```bash
pip install -r requirements.txt
```

## Project Structure

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
├── utils                                   - Package for utilities.
|   ├── plotter.py                          - Module for plotting visual utilities and results.
```

## Architectural Modules

- ### **Data Loader**

	- The "data_loader" package loads the experiment specific dataset from the Keras library or from a specified path.
	- Each dataset should have a separate module implementation.

- ### **Model**

	- It implements the ConvNet architecture as per the requirements of the experiment.
	- This package holds one specific module for each experiment that designs the model's architecture.

- ### **Mains**
	- The main package wraps the execution flow of the experiment.
	- Depending on the requirement it calls the specific member methods from different modules to perform a specific operation.
	- It's class diagram is illustrated in the following figure: </br>
	![Mains](./resources/class_diagram.png)