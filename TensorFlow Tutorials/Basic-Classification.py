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
# plt.show()

# Scale the 0-255 values to 0-1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Show a sample image after processing
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
# plt.show()

# Display first 25 images from training set with class names
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# plt.show()

# BUILD THE MODEL
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Converts the layer from 2d (28x28) to 1d 784)
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)  # Returns an array of 10 probability scores that sum to 1
])

# Compiling the model
model.compile(optimizer=tf.train.AdamOptimizer(),
              # How the model is updated based on the data it sees and its loss function
              loss='sparse_categorical_crossentropy',  # Measures how accurate the model is during training
              metrics=['accuracy'])  # Monitors the training and testing steps, in this case accuracy

# TRAINING THE MODEL
model.fit(train_images, train_labels, epochs=5, verbose=2)  # verbose 0 - no log, 1 - full log, 2- summary

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy: ', test_acc)

# MAKE PREDICTIONS
predictions = model.predict(test_images)
# Select the highest confidence score
print("Predicted Label: ", np.argmax(predictions[0]))
print("Test Label: ", test_labels[0])

# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label],
                                class_names[true_label]),
               color=color)
#plt.show()

# Grab image from the test dataset
img = test_images[0]
print("Test Image Shape: ", img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print("Test Image Shape: ", img.shape)

predictions = model.predict(img)

print("Prediction: ", predictions)

prediction = predictions[0]

print(np.argmax(prediction))