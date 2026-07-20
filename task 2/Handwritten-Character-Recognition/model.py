"""Handwritten Character Recognition using a CNN.

This script loads the MNIST dataset, preprocesses the images, trains a
convolutional neural network, evaluates its performance, saves the trained
model, and creates a sample prediction figure.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Keep TensorFlow logs concise for a cleaner beginner project experience.
tf.get_logger().setLevel("ERROR")


# Project paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
MODEL_PATH = Path("handwritten_character_model.h5")
PLOT_PATH = OUTPUT_DIR / "sample_predictions.png"
RANDOM_STATE = 42


def load_and_prepare_data():
    """Load MNIST and prepare train/test arrays for a CNN."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values to the [0, 1] range.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape to add the channel dimension expected by CNNs.
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def build_cnn_model():
    """Create a simple CNN for handwritten digit classification."""
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_sample_predictions(model, x_test, y_test):
    """Save a figure showing sample predictions from the trained model."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    sample_images = x_test[:9]
    sample_labels = y_test[:9]
    predictions = model.predict(sample_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10, 10))
    for index in range(9):
        # OpenCV is imported as required and used here for a small image resize
        # operation before plotting, demonstrating basic image-processing usage.
        image = sample_images[index].squeeze()
        resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_NEAREST)

        axis = plt.subplot(3, 3, index + 1)
        axis.imshow(resized_image, cmap="gray")
        axis.set_title(f"Pred: {predicted_labels[index]} | True: {sample_labels[index]}")
        axis.axis("off")

    plt.suptitle("Sample Handwritten Digit Predictions")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Load and prepare the dataset.
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data()

    # Build the CNN model.
    model = build_cnn_model()

    # Train the model.
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=128,
        verbose=1,
    )

    # Evaluate on the held-out test set.
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    train_accuracy = history.history["accuracy"][-1]

    print("\nTraining completed")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save the trained model in HDF5 format.
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    # Create and save sample predictions.
    save_sample_predictions(model, x_test, y_test)
    print(f"Sample predictions saved to: {PLOT_PATH}")
