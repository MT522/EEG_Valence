# EEG Feature Selection and Classification with MLP in PyTorch

This project performs feature engineering, selection, and classification on EEG data using a Multi-Layer Perceptron (MLP) implemented in PyTorch.

## 📁 Project Structure

- `Project_data.mat`: Contains raw EEG training/testing data and sampling frequency.
- `.npy` files: Pre-extracted EEG features.
- Main script: Preprocesses features, selects the best ones, trains and evaluates an MLP model using 5-fold cross-validation.

## ⚙️ Features Used

Eight different types of features are concatenated and scaled:

1. Variance
2. Amplitude Histogram
3. AR Model Coefficients
4. Cross-Correlation
5. Maximum Frequency
6. Mean Frequency
7. Median Frequency
8. Relative Energy

The top 50 features are selected using a custom **Fisher Score** method.

## 🧠 Model Architecture

A simple MLP with:

- Input Layer: 50 selected features
- Hidden Layer 1: 32 neurons + ReLU
- Hidden Layer 2: 16 neurons + ReLU
- Output Layer: 1 neuron + Sigmoid

Binary Cross-Entropy Loss is used for classification.

## 🚀 Training

- Optimizer: Adam
- Batch size: 10
- Epochs: 10
- Cross-validation: 5-Fold

Model is trained and evaluated on each fold, and accuracy is printed per fold.

## Usage

You must first run the cells in the notebook to generate the npy data files. Afterwards, you may run the script `pipeline.py`.
