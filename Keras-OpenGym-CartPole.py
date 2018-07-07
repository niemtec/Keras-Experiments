from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import model_from_json
from random import randint

# Neural Net for Deep Q Learning
# Sequential() creates the foundation of the layers.
model = Sequential()
# 'Dense' is the basic form of a neural network layer
# Input Layer of state size(4) and Hidden Layer with 24 nodes
model.add(Dense(24, input_dim=model.state_size, activation='relu'))
# Hidden layer with 24 nodes
model.add(Dense(24, activation='relu'))
# Output Layer with # of actions: 2 nodes (left, right)
model.add(Dense(model.action_size, activation='linear'))
# Create the model based on the information above
model.compile(loss='mse',
              optimizer=Adam(lr=model.learning_rate))
