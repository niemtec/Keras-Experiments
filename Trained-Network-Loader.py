import numpy as np
from matplotlib import pyplot as plt

np.random.seed(123)  # for reproducibility

# Importing the sequential model from Keras
from keras.models import Sequential

# Importing core layers from Keras
from keras.layers import Dense, Dropout, Activation, Flatten

# Importing CNN layers from Keras - they will help efficiently train on image data
from keras.layers import Convolution2D, MaxPooling2D

# Importing utilities
from keras.utils import np_utils

# Import mnist dataset
from keras.datasets import mnist

from keras.models import model_from_json


# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test = X_test[600:601]
y_test = y_test[600:601]

plt.imshow(X_test[0])
plt.show()
print(y_test[0])

# Transforming the dataset to show depth (1 colour)
X_test = X_test[0].reshape(X_test.shape[0], 28, 28, 1)

# Convert data type to float32 and normalise data values to the range [0, 1]
X_test = X_test.astype('float32')
X_test /= 255

# Convert 1D class arrays to 10D class matrices (10 numbers)
Y_test = np_utils.to_categorical(y_test, 10)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
# Evaluate the model on test data
#score = model.evaluate(X_test, Y_test, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))  # Print the model metrics

pr = model.predict_classes(X_test)
print(pr)