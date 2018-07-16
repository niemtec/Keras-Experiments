# TensorFlow and tf.keras
import tensorflow as tf
from keras.legacy import layers
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

# Using the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
# Load the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names for the dataset (not natively included)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# PRE-PROCESSING THE DATA
# Show a sample image before processing
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()

# Scale the 0-255 values to 0-1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Show a sample image after processing
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()

# Display first 25 images from training set with class names
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# BUILD THE MODEL
# Typical model for a fully-connected network (perceptron)
model = keras.Sequential()
# Add a densly-connected layer with 64 units
model.add(keras.layers.Dense(64, activation='relu'))
# Add another layer
model.add(keras.layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units
model.add(keras.layers.Dense(10, activation='softmax'))

# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')
# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))
# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01))
# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')
# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=keras.initializers.constant(2.0))