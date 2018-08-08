# FlowerPower

FlowerPower is a collection of ResNet-based neural network architectures and accompanying utility functions. 

## Setup

### Requirements

The code makes extensive use of many different packages, that's why the Python package manager Anaconda was chosen to simplify the setup process. You can download Anaconda from [here](https://www.anaconda.com/download/#linux). Please be sure to confirm Anaconda to your `.bashrc` to be able to start the program from the command line.

### Creating an Environment

The individual packages are not listed here, as all requirements are contained in the `requirements.txt` provided in this repository. After you have installed Anaconda, create a new Python environment using the requirements file:
```
conda create --name myenv --file requirements.txt
```
You can name the environment in any way you like. To be able to run code contained in this repository, you have to activate the environment:
```
source activate myenv
```
You are now ready to run the provided Python scripts.

## Code Structure
