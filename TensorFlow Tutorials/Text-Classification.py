# Classification of movie review sentiment (pos/neg) by analysing review text
# Binary Classification Example

import tensorflow as tf
from tensorflow import keras
import numpy as np
print(tf.__version__)

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) # keeps top 10000 words in the training data

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
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])