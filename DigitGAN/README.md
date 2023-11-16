# Instructions

## 1. Hyperparameters and Models

Adjust values in settings.cfg**
Tune model parameters in TorchModels.py**
Change model hidden layer size where model instantiation occurs in the trainer.py files

## 2. Training

Option 1: TorchTrainerDCGAN.py, trains a deep convolutional model\_\_
Option 2: TorchTrainer.py, trains a deep neural network model

## 3. Viewing Results

First, rename the best epoch model to GenModel.pth, delete other models if you wish\_\_
Next set the filename variable in ViewResults.py to the model folder, and run
