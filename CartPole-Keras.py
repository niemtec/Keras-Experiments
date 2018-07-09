import gym
import random
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup = "cartpole_weight.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        # Learning Rate indicates how much NN learns from the loss between the target and the prediction in each iteration
        self.learning_rate = 0.001
        # Gamma is used to calculate the future discounted reward
        self.gamma = 0.95
        # Initially actions chosen at random until agent gains experience
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        # Decreases the number of random explorations as agent learns
        self.exploration_decay = 0.995
        self.brain = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model using the Sequential framework from Keras
        model = Sequential()
        # Add a basic (dense) layer
        # Input size is 4 (four objects returned from the environment)
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # Hidden layer with 24 nodes
        model.add(Dense(24, activation='relu'))
        # Output layer with two actions: 2 nodes (left, right move)
        model.add(Dense(self.action_size, activation='linear'))
        # Create the model based on the configuration above
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        # Used for backups of the weights at each stage
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    # Saves the weight configuration
    def save_model(self):
        self.brain.save(self.weight_backup)

    # Performs an action randomly or using NN prediction from current state
    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        # Predict reward of current state based on the data available
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    # Save observation from environment to memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        # The number of games to play for training
        self.episodes = 10000
        self.env = gym.make('CartPole-v1')

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.state_size, self.action_size)

    def run(self):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

                done = False
                index = 0
                while not done:
                    self.env.render()

                    action = self.agent.act(state)

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()


if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()
