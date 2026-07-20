# Handwritten Character Recognition

## Project Overview

Handwritten Character Recognition is a beginner-friendly deep learning project that classifies handwritten digits using a Convolutional Neural Network (CNN). The project demonstrates a complete image classification workflow with preprocessing, training, evaluation, and visualization.

## Objective

The objective of this project is to identify handwritten characters or alphabets from image data using deep learning. This version uses the MNIST digit dataset as the main training source, with EMNIST suggested as a natural extension for alphabet recognition.

## Features

- Loads the MNIST dataset using Keras
- Normalizes pixel values for neural network training
- Reshapes images for CNN input
- Trains a convolutional neural network
- Evaluates the model on the test set
- Prints training and test accuracy
- Generates sample handwritten digit predictions
- Saves the trained model as an HDF5 file
- Saves a prediction figure in the outputs folder

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn as a supporting dependency in the project environment

## Dataset Description

This project uses the MNIST dataset, a standard handwritten digit dataset provided by Keras through `keras.datasets.mnist`. It contains grayscale images of handwritten digits from 0 to 9.

For alphabet recognition, EMNIST can be used as an extension because it includes handwritten letters and characters, making it a strong follow-up dataset for broader character classification.

## CNN Architecture

The model uses a simple CNN architecture suitable for beginners:

- Input layer for 28x28 grayscale images
- Convolution layer with ReLU activation
- Max pooling layer
- Second convolution layer with ReLU activation
- Second max pooling layer
- Flatten layer
- Dense hidden layer
- Dropout layer
- Final softmax output layer for 10 classes

## Training Process

1. Load the MNIST dataset.
2. Normalize image values to the range 0 to 1.
3. Reshape images to include a channel dimension.
4. Train the CNN using categorical labels.
5. Validate the model during training.
6. Evaluate the final model on the test set.
7. Save the trained model and sample prediction plot.

## Results

After training, the script prints:

- Training accuracy
- Test accuracy

It also saves a visual result showing sample predictions compared with true labels.

## Folder Structure

```text
Handwritten-Character-Recognition/
├── data/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── model.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── project_structure.txt
```

## Installation

1. Clone or open the project folder.
2. Create a virtual environment.
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the training script with:

```bash
python model.py
```

After execution, the trained model will be saved as `handwritten_character_model.h5`, and the sample predictions image will be saved to `outputs/sample_predictions.png`.

## Future Improvements

- Extend the project from digits to alphabet recognition using EMNIST
- Improve the CNN architecture for higher accuracy
- Add data augmentation
- Add training history plots
- Build a small user interface for live handwritten input predictions

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
