from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import model_from_json
from random import randint

# Generate a random number to select from the dataset
random = randint(0, 10000)

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test = X_test[random:random + 1]
y_test = y_test[random:random + 1]
plt.imshow(X_test[0])
plt.show()

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
# score = model.evaluate(X_test, Y_test, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))  # Print the model metrics

pr = model.predict_classes(X_test)
print("Number used in this test: ", y_test[0])
print("Network detected: ", pr[0])
