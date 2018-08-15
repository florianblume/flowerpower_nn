# FlowerPower

FlowerPower is a collection of ResNet-based neural network architectures and accompanying utility functions. 

## Setup

### Requirements

The code makes extensive use of many different packages, that's why the Python package manager Anaconda was chosen to simplify the setup process. You can download Anaconda from [here](https://www.anaconda.com/download/#linux). Please be sure to confirm Anaconda to your `.bashrc` to be able to start the program from the command line.

### Creating an Environment

The individual packages are not listed here, as all requirements are contained in the `requirements.txt` provided in this repository. After you have installed Anaconda, create a new Python environment using the requirements file. But first, you have to add the forge channel and manually install two packages.
```
conda install -c anaconda libxml2 
conda install -c conda-forge libiconv 
conda config --add channels conda-forge 
conda create --name myenv --file requirements.txt
```
You can name the environment in any way you like. To be able to run code contained in this repository, you have to activate the environment:
```
source activate myenv
```
You are now ready to run the provided Python scripts.

## Network

### Training

To run network trainnig, use the `training.py` script and pass the path to the training config. You can use the example config to see what it has to look like. It is only necessary to define multiple learning rates, epochs, and layers to train if using the SGD optimizer (edit the model for this). Adam adjusts its learning rates automatically. You can train different architectures by setting the "MODEL" value in the config. Have a look at the models folder to see what models are available.

## Inference

To run network inference you have two options: use `inference_raw.py` to obtain the actual object coordinate predictions as a tiff file. To directly predict poses use `inference_pos.py`. Have a look at the example configurations what they should look like. You have to set the correct model which matches the weights file you are using.

## Utility

There are also some utility scripts that should be selfexplanatory.
