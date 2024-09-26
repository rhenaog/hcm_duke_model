# HCM Model Training

This repository contains the Python (and Pytorch) source code for training and testing the model for HCM prediction described in [1].

## train_hcm_model.py
This file is used for training the HCM model. The base model used is r2plus1d model which can be loaded from torchvision.models. Two datasets are parsed: training and validation. Training dataset is used to train the model, and validation dataset is used to infer how well the model performs on non-training dataset. The best model is saved based on the metric from the validation set.

When running the code, the code will show a progress bar for each epoch and outputs training and validation accuracy and AUC. The training will stop based on early stopping method and the best performing model will be saved accordingly.

## data_loader_hcm.py
This file is used to load the data for training and validation. The file takes in two csv files which includes the directory for the data and the view probabilities (SAX, AP2, AP3, AP4). The data is resized to (112, 112) and the background information are removed through masking. For training, a random set of 16 consecutive frames are extracted for each epoch. For validation and test set, all frames are extracted and aggregated for prediction.

## utils.py
This file contains functions for saving and loading the model (.pt format).

## test_hcm_model.ipynb
This is a Jupyter notebook file used for making inference on test dataset. The code will load the model based on training and validation and make predictions on the test dataset (or whichever dataset the user defines). Since each subject may have more than one recordings, the final output AUC is given by aggregating all the probabilities predicted from the model for each subject.

## Reference
[1] Authors. Title. Journal. Year.
