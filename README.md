# Parkinson’s Disease Detection Using Biomedical Voice Features

## Team Members

- Charmilkumar Vijaykumar Patel
- Yunyang Zhang

## Project Description

This project focuses on Parkinson’s disease detection using biomedical voice measurement features. Parkinson’s disease is a progressive neurological disorder that can affect movement, coordination, and speech. Since voice patterns may change in people with Parkinson’s disease, biomedical voice data can be used as a helpful source for prediction.

The goal of this project is to build a Python-based program that loads a Parkinson’s voice dataset, prepares the data, trains a machine learning model, evaluates the model performance, and reports the prediction results.

This project demonstrates Python programming skills such as data input/output, object-oriented programming, exception handling, use of external libraries, docstrings, comments, and advanced Python features.

## Dataset

The dataset used in this project is a Parkinson’s disease biomedical voice dataset. The dataset is stored in the data folder.

The target column is status. It indicates whether a voice sample is related to Parkinson’s disease. The other columns contain biomedical voice measurements such as frequency, jitter, shimmer, and noise-related features.

## Project Overview

The program performs the following steps:

1. Load the Parkinson’s disease voice dataset from a CSV file.
2. Check and prepare the dataset.
3. Separate the input features and target label.
4. Split the dataset into training and testing sets.
5. Train a machine learning classification model.
6. Predict the testing data labels.
7. Evaluate the model using accuracy score, confusion matrix, and classification report.
8. Display the final results.

## Dependencies / Libraries

This project uses the following Python libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- time

To install the required libraries, run:

pip install pandas numpy matplotlib scikit-learn

## File Structure

AAI-551-WS/
- data/
- src/
- README.md
- parkinsons_detection.ipynb

## File / Folder Description

### data/

This folder contains the Parkinson’s disease biomedical voice dataset used for this project.

### src/

This folder contains Python source files used for dataset loading, model training, evaluation, and helper functions.

### parkinsons_detection.ipynb

This is the main Jupyter Notebook for running the full project. It loads the dataset, trains the model, evaluates the results, and displays the final output.

### README.md

This file explains the project purpose, dataset, dependencies, file structure, running instructions, and team member contributions.

## How to Run the Program

### Step 1: Download or clone the repository

Download or clone this repository from GitHub.

### Step 2: Install required libraries

Run the following command:

pip install pandas numpy matplotlib scikit-learn

### Step 3: Open the Jupyter Notebook

Open the file:

parkinsons_detection.ipynb

### Step 4: Run all cells

Run the notebook from top to bottom. The notebook will load the dataset, train the model, evaluate the prediction results, and display the final model performance.

## Python Requirements Satisfied

This project is designed to satisfy the required Python programming features, including:

- Classes and object-oriented programming
- Functions
- Data input/output using a CSV file
- Exception handling for missing files and invalid data
- Loops and conditional statements
- Mutable and immutable data types
- Docstrings and meaningful comments
- Use of external libraries
- Operator overloading
- List comprehension
- Set operations
- Generator function
- Built-in Python module

## Exception Handling

The program includes exception handling for common errors, such as:

- Missing dataset file
- Missing required columns
- Invalid prediction before model training
- Invalid or empty dataset input

These checks help make the program more reliable and easier to understand.

## Model Evaluation

The model is evaluated using:

- Accuracy score
- Confusion matrix
- Classification report

These evaluation methods help show how well the model predicts Parkinson’s disease status based on biomedical voice features.

## Main Contributions of Each Team Member

### Charmilkumar Vijaykumar Patel



### Yunyang Zhang



## Notes

This project is for educational purposes. The model is not intended for real medical diagnosis. It only demonstrates how biomedical voice features can be used in a Python-based machine learning project.
