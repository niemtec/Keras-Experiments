# Classification of movie review sentiment (pos/neg) by analysing review text
# Binary Classification Example

import tensorflow as tf
from tensorflow import keras
import numpy as np
print(tf.__version__)

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
