# MNIST Digit Classification with Convolutional Neural Networks

This project is an implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits using the MNIST dataset. The CNN architecture is built using TensorFlow and Keras.

## Overview

In this project, we train a CNN model to classify images of handwritten digits from 0 to 9. The model architecture consists of multiple convolutional and pooling layers followed by fully connected layers. We use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

## Getting Started

### Prerequisites

To run this project, you need:

- Python
- TensorFlow

You can install the required Python packages using pip:

```bash
pip install tensorflow 
```

### Usage

1. Clone the repository:

```bash
git clone <repository_url>
cd DigitClassifier.ipynb
```

2. Run the Jupyter Notebook:

```bash
jupyter notebook DigitClassifier.ipynb
```

### File Structure

- `DigitClassifier.ipynb`: Jupyter Notebook containing the code for training the CNN model and making predictions.
- `images/`: Directory containing sample images for testing the model.

## Results

After training the model, we achieved an accuracy of 99% on the test set.
