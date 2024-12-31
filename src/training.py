import os
import tensorflow as tf
from agentic_ai import AgenticAI
from tensorflow.keras.datasets import mnist
import numpy as np

def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = np.expand_dims(train_images, -1).astype('float32') / 255.0
    test_images = np.expand_dims(test_images, -1).astype('float32') / 255.0
    return train_images, train_labels, test_images, test_labels

def main():
    train_images, train_labels, test_images, test_labels = load_data()

    model = AgenticAI()
    model.train(train_images, train_labels, epochs=5)

    predictions = model.predict(test_images)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
