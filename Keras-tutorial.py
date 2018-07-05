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

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Shows the shape of the dataset (demo only)
# print (X_train.shape)
# (60000, 28, 28)

# Plotting an image for fun
plt.imshow(X_train[0])
plt.show()

# Transforming the dataset to show depth (1 colour)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Convert data type to float32 and normalise data values to the range [0, 1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert 1D class arrays to 10D class matrices (10 numbers)
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Declaring a sequential model format
model = Sequential()

# Declare input layer
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28, 1)))

# Adding more layers to the network
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Adding a fully connected layer and then the output layer
model.add(Flatten())  # weights are being flattened (1D) before passing them to the Dense layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# Final layer has output size of 10, corresponding to 10 digits
model.add(Dense(10, activation='softmax'))

# Compile the model and declare a loss function
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# To fit the model we have to declare the batch size and number of epochs to train for, then pass in the training data
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)

# Evaluate the model on test data
score = model.evaluate(X_test, Y_test, verbose=0)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")