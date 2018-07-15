# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

# Using the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
# Load the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
