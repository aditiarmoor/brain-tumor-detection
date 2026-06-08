# Brain Tumor Detection Using Deep Learning

## Overview

This project is a Brain Tumor Detection System developed using Deep Learning and MRI brain scan images. The application uses a Convolutional Neural Network (CNN) model to classify MRI images and predict whether a brain tumor is present.

## Features

* MRI Image Upload
* Brain Tumor Detection
* Deep Learning-based Classification
* User-Friendly Streamlit Interface
* Fast and Accurate Predictions

## Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* OpenCV
* NumPy
* SQLite

## Project Structure

```text
Brain_Tumor_Detection/
│
├── app.py
├── users.db
├── yes/
├── no/
├── brain_tumor_dataset/
└── README.md
```

## Model File

The trained model file (`brain_tumor_model.h5`) is not included in this repository because it exceeds GitHub's file size limit.

Download the model from:

https://drive.google.com/file/d/15jc44htWqn1kSejUsR5wrnsUGGe_J8df/view?usp=sharing

After downloading, place the file in the project root directory.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

The dataset consists of MRI brain scan images categorized into:

* Tumor (Yes)
* No Tumor (No)

## Future Enhancements

* Multi-class Brain Tumor Classification
* Improved Model Accuracy
* Cloud Deployment
* Doctor Recommendation System

## Author

Aditi Armoor
