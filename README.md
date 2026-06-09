# Pneumonia Detection Web Application

This repository contains a clean and simple Deep Learning web application that detects whether a given chest X-ray indicates **Viral Pneumonia**, **Bacterial Pneumonia**, or is **Normal**. 

The underlying model is a Hybrid VGG-DenseNet Convolutional Neural Network built with TensorFlow & Keras. The web interface is built entirely in Python using [Gradio](https://gradio.app/).

## Features
* **Simple Uploads**: Drag and drop or browse to upload a chest X-ray image.
* **Automatic Pre-processing**: The app automatically handles color conversions, resizing the image to the required `224x224` resolution, and normalizing pixel values between `0-1`.
* **Multi-Class Prediction**: Displays classification results for Viral, Bacterial, or Normal X-Rays.
* **Confidence Score**: Outputs the exact probability percentage computed by the model.

## Prerequisites
To run this project, make sure you have:
* Python 3.8+ installed.
* An active python environment / virtual environment with sufficient storage space (TensorFlow is large).

## Installation Details
We recommend using a Python Virtual Environment (`venv`) to avoid path length limit errors and missing packages on Windows, which helps isolate this project's packages.

### 1. Setup the Virtual Environment
Navigate to the project directory in your terminal:
```bash
# Create the virtual environment called 'venv'
python -m venv venv
```

### 2. Install Dependencies
Activate the environment and install everything listed in `requirements.txt`:
```bash
# Using the pip inside the newly created virtual environment
.\venv\Scripts\pip.exe install -r requirements.txt
```
> **Note on Storage:** Installing TensorFlow requires approximately 3 to 4 GB of free storage space on your hard drive. If you run into a `[Errno 28] No space left on device` error, clear some files and run the command again.

## Usage
Once installed, run the python script using your virtual environment directly:
```bash
.\venv\Scripts\python.exe app.py
```

The terminal will print out a local server link, usually:
`Running on local URL:  http://127.0.0.1:7860/`

Hold `Ctrl` and click that link (or copy and paste it into your browser) to open the web interface. 

## Model Configuration Note
Inside `app.py`, under the prediction processing lines, you will find a classes list:
```python
classes = ["Normal", "Bacterial", "Viral"]
```
This maps the Neural Network output indexes (`0`, `1`, `2`) to a human-readable string. If during your model's training process the alphabetized order of the dataset folders was different, you will need to adjust the names in that array inside `app.py` to match!
