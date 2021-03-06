# Classification of movie review sentiment (pos/neg) by analysing review text
# Binary Classification Example

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

print(tf.__version__)

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)  # keeps top 10000 words in the training data

# EXPLORE THE DATA
# Each example is an array of integers representing the words of the movie review.
# Each label is an integer value of either 0 or 1, where 0 is a negative review and 1 is a positive review.

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# The text of reviews have been converted to integers, where each integer represents a specific word in a dictionary.
print(train_data[0])

# Show number of words in first and second reviews
print(len(train_data[0]), len(train_data[1]))

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# Example of using a decoder to see the review
print(decode_review(train_data[0]))

# PREPARING THE DATA
# The reviews—the arrays of integers—must be converted to tensors before fed into the neural network.
# Reviews must be of same length - they will be padded to standardise their lengths
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print("Review length after padding: ", len(train_data[0]), len(train_data[1]))

# Show first padded review
print(train_data[0])

# BUILD THE MODEL
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
# L1: Takes the integer-encoded vocabulary and looks up embedding vector for each word index
# The vectors are learned as the model trains
model.add(keras.layers.Embedding(vocab_size, 16))
# Returns a fixed-length output vector for each example (by averaging over the sequence dimension)
# Allows model to handle input of variable length in the simplest way possible
model.add(keras.layers.GlobalAveragePooling1D())
# Fixed-length output vector piped through a fully-connected (dense) layer with 16 hidden units
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# Densly connected layer with a single output node
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# The model has two hidden layers giving it more freedom to learn
# More hidden layers can be used to learn more complex representations (expensive)
# Too many layers can cause network to learn unwanted patterns which causes overfitting


# Loss function and optimiser
# Since this is a binary classification problem, we use binary_crossentropy loss function (good for probabilities)
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Validation set for checking accuracy of model on unseen data
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# TRAINING THE MODEL
# Training for 20 epoch in mini-batches of 512 samples (20 iterations over all samples)
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# EVALUATE THE MODEL
results = model.evaluate(test_data, test_labels)
print(results)

# CREATE GRAPHS OF ACCURACY
# Using the history dictionary to plot the results
history_dict = history.history
history_dict.keys()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "Bo" is for "Blue Dot"
plt.plot(epochs, loss, 'bo', label='Training Loss')
# b is for "Solid Blue Line"
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Training-and-validation-loss.png')
plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Training-and-validation-accuracy.png')
plt.show()