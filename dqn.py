import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import tensorflow as tf

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=2000, verbose=0):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # initial exploration rate
        self.initial_epsilon = epsilon  # store initial epsilon to reset later
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Neural Net for Deep-Q learning Model."""
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copy weights from model to target_model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experiences in replay memory."""
        # Ensure states are 2D arrays when remembered
        self.memory.append((np.array(state).reshape(1, -1), action, reward, np.array(next_state).reshape(1, -1), done))

    def act(self, state):
        """Return action based on the current state."""
        state = np.array(state).reshape(1, -1)  # Ensure state is a 2D array
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Train the model using randomly sampled experiences from the memory."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            next_state = np.array(next_state).reshape(1, -1)  # Ensure next_state is a 2D array
            target = reward if done else reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            state = np.array(state).reshape(1, -1)  # Ensure state is a 2D array
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0, batch_size=1)  # Using 1 since each state is a separate sample
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path_to_weights):
        """Load saved model weights."""
        self.model.load_weights(path_to_weights)

    def save(self, path_to_weights):
        """Save model weights."""
        self.model.save_weights(path_to_weights)

    def reset(self):
        """Reset the agent state between episodes."""
        self.memory.clear()
        # Reset epsilon to initial value
        self.epsilon = self.initial_epsilon
        # Optionally reset model weights
        # self.model.set_weights(self.target_model.get_weights())
