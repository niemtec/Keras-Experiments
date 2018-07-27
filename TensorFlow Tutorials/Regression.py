# REGRESSION
# Aim is to predict the output of a continuous value such as probability

import tensorflow as tf
import pandas as pd
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

# Download and shuffle the dataset
boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

order = np.argsort(np.random.random(train_labels.shape))
training_data = train_data[order]
train_labels = train_labels[order]

print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))  # 102 examples, 13 features

# Each feature uses a different scale:
print(train_data[0])  # Display sample features to show different scales

# Display the first few rows of the dataset as a table
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
df.head()

# Labels
print(train_labels[0:10])  # Display labels (in thousands of dollars)

# Normalise Features
# Normalising features is recommended for features that use different scales and ranges,
# for each feature subtract the mean of the feature and divide by standard deviation

# Test data is not used when calculating the mean and std
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])  # First normalised training sample


# Although the model might converge without feature normalization, it makes training more difficult,
# it makes the resulting model more dependant on the choice of units used in the input.

# Create the model
# Using a Sequential model with two densly connected hidden layers and output layer returning a single value
# Wrapped in a function for later reuse
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimiser = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimiser,
                  metrics=['mae'])

    return model


model = build_model()
model.summary()


# Train the Model
# Display training progress my printing a '.' for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 500

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose = 0,
                    callbacks=[PrintDot()])


# Visualise model training process using stats stored in history object
# -> used to determine how long to train before the model stops making progress
def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val Loss')
    plt.legend()
    plt.ylim([0,5])

plot_history(history)

# Make model.fit stop training when validation doesn't improve
model = build_model()
# Practice parameter is the number of epoch to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print()
print("Testing set Mean Absolute Error: ${:7.2f}".format(mae*1000))

# Predict
test_predictions = model.predict(test_data).flatten()
print(test_predictions)
