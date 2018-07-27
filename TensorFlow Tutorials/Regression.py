# REGRESSION
# Aim is to predict the output of a continuous value such as probability

import tensorflow as tf
import pandas as pd
from tensorflow import keras

import numpy as np

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
