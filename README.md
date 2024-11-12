# Friendship Sloop Detector Model

## Overview

This repository contains a Jupyter notebook and a Streamlit app that demonstrate how to build, train, and deploy a deep learning model to detect Friendship Sloops in images. The model is based on the ResNet50 architecture and uses data augmentation techniques to improve generalization. The notebook includes steps for data preprocessing, model definition, training, and evaluation, while the Python script provides a Streamlit-based web application for model inference.

## Prerequisites

Before running the notebook or the Python script, ensure you have the following prerequisites:

- Python 3.6 or higher
- TensorFlow 2.x
- Keras
- OpenCV
- Matplotlib
- dotenv
- Streamlit
- scipy

## Required Directory Structre

```bash
├── test
│   ├── friendship_sloop
│   └── no_friendship_sloop
└── train
    ├── friendship_sloop
    └── no_friendship_sloop
```

## Setup Instructions

### 1. Clone the Repository
Clone the repository containing this notebook and navigate to the directory. Be sure to create the virtual environment.
```bash
git clone <repository_url>
cd <repository_directory>
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install the required Python packages using pip.
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a .env file in the root directory and define the following environment variables:
```bash
DATA_DIR=path/to/your/data
MODEL_PATH=path/to/save/model
```

### 4. Run the Notebook
Open the notebook in Jupyter and run all cells. Please note: This notebook was developed in VSCode. To use this Notebook in the Jupyter environment, use the following command:
```bash
jupyter notebook Build_Friendship_Sloop_Detector_Model.ipynb
```

### 5. Run the Streamlit App
Run the Streamlit app to use the trained model for inference.
```bash
streamlit run Friendship_Sloop_Detector.py
```

## Notebook Sections

### Import Libraries
Imports the necessary libraries for data preprocessing, model building, and environment management.

### Load Environment Variables
Loads environment variables from a .env file using the dotenv library.

### Define Configuration Class
Defines the CFG class to store configuration settings such as data directory, batch size, number of epochs, and model path.

### Data Preprocessing
Sets up data augmentation and preprocessing using ImageDataGenerator. Defines training and validation data generators.

### Model Definition
Defines the model architecture using the ResNet50 base model with custom classification layers. Freezes the base model's layers.

### Model Training
Trains the model using the training and validation data generators. Monitors the training process using validation accuracy and loss.

### Model Evaluation
Evaluates the model's performance on the validation dataset and prints the validation accuracy and loss.

### Save the Model
Saves the trained model to the specified path.

## Python Script: Friendship_Sloop_Detector.py

The Friendship_Sloop_Detector.py script provides a Streamlit-based web application for detecting Friendship Sloops in uploaded images. Below is an overview of the script:

## Script Overview

### Import Libraries:

Imports necessary libraries for model loading, image preprocessing, and Streamlit app creation.

### Load Environment Variables:

Loads environment variables from a .env file using the dotenv library.

### Define Configuration Class:

Defines the CFG class to store configuration settings such as data directory, batch size, number of epochs, and model path.

### Load the Model:

Loads the trained model from the specified path.

### Set Up Streamlit App:

Sets up the Streamlit app title and file uploader for image input.

### Image Preprocessing Function:

Defines a function to preprocess uploaded images to match the model input size.

## Conclusion
This repository provides a comprehensive guide to building, training, and deploying a Friendship Sloop detector model using TensorFlow, Keras, and Streamlit. By following the steps outlined, you can preprocess your data, define a robust model architecture, train the model, evaluate its performance, and deploy it as a web application. The use of data augmentation and transfer learning techniques helps improve the model's generalization and accuracy.