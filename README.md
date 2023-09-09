# Aritificial_Intel_Assignment
The purpose of the assignment is to create a CNN and perform hyperparameter tuning for hand digit classification
The data set can be obtained from kaggle from the following link https://www.kaggle.com/datasets/koryakinp/fingers

# Image Classification with ResNet18

## Overview

This repository contains a Jupyter Notebook (`A3_Q4.ipynb`) designed for image classification using a ResNet18 deep learning model. The notebook includes various sections for data preprocessing, model training, testing, and more. It was originally created in a Google Colaboratory environment.

## Table of Contents

- [Dataset Preparation](#dataset-preparation)
- [Importing Libraries](#importing-libraries)
- [Setting Up the Environment](#setting-up-the-environment)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Inference on Sample Images](#inference-on-sample-images)

## Dataset Preparation

Before running the notebook, you should have your image dataset ready. The code includes a section to unzip the dataset from a file named `archive.zip`. Make sure to place your dataset in the appropriate location or update the path accordingly.

## Importing Libraries

The notebook begins by importing essential Python libraries for image processing, deep learning, and data visualization. Ensure that you have the required packages installed before running the notebook.

## Setting Up the Environment

The notebook sets up the computing environment by determining whether a GPU (CUDA) is available and choosing the appropriate device. It also defines a label map for the classes in your dataset.

## Data Preprocessing

This section prepares the dataset for training and testing. It includes data augmentation techniques such as resizing, rotation, color jitter, and normalization. Images are organized into folders based on class labels.

## Model Training

The notebook trains a ResNet18 deep learning model using the prepared dataset. You can customize hyperparameters such as learning rate, batch size, and optimizer (Adam or SGD) to suit your specific dataset and problem. Training progress is displayed, and model checkpoints are saved for future use.

## Model Evaluation

After training, the notebook evaluates the performance of the trained model using various metrics, including accuracy, confusion matrix, and a classification report. This section provides insights into how well the model is performing on your dataset.

## Inference on Sample Images

The notebook concludes by demonstrating how to use the trained models for inference. You can pass sample images through the models, and the predicted class labels are displayed.

## Usage

To use this notebook, follow these steps:

1. Clone or download this repository to your local machine.

2. Ensure you have the necessary Python packages installed. You can use `pip` or `conda` to install them as needed.

3. Prepare your image dataset and place it in the appropriate location or update the dataset path in the notebook.

4. Open and run the Jupyter Notebook (`A3_Q4.ipynb`) using Jupyter Notebook or JupyterLab.

5. Customize hyperparameters and training settings as required for your dataset.

6. Execute the code cells step by step, following the instructions in the notebook.

7. Train and test the models, and evaluate their performance on your dataset.

8. Use the provided code for inference on your own images if needed.

9. Save the trained models for future use using the provided code for model checkpointing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The notebook uses the ResNet18 architecture and weights from the [PyTorch torchvision library](https://pytorch.org/vision/stable/models.html).

