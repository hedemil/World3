import numpy as np
import random
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import os
import tensorflow as tf

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=2000, verbose=0, model_path=None):
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
        if model_path and os.path.isfile(model_path):
            self.model = load_model(model_path)
        else:
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

    # def remember(self, state, action, reward, next_state, done):
    #     """Store experiences in replay memory."""
    #     try:
    #         # Ensure states are 2D arrays when remembered
    #         self.memory.append((np.array(state).reshape(1, -1), action, reward, np.array(next_state).reshape(1, -1), done))
    #     except Exception as ex:
    #         print(f"An exception occurred during agent remember: {ex}")
    
    # With StateNormalizer
    def remember(self, state, action, reward, next_state, done):
        """Store experiences in replay memory."""
        try:
            # Directly append the states assuming they are already the correct numpy arrays
            self.memory.append((state, action, reward, next_state, done))
        except Exception as ex:
            print(f"An exception occurred during agent remember: {ex}")

    def act(self, state):
        """Return action based on the current state."""
        try:
            state = np.array(state).reshape(1, -1)  # Ensure state is a 2D array
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])
        except Exception as ex:
            print(f"An exception occurred during action prediction: {ex}")
            # Handle the exception, for example by taking a random action
            return random.randrange(self.action_size)

    
    # With StateNormalizer    
    # def replay(self, batch_size):
    #     """Train the model using randomly sampled experiences from the memory."""
    #     if len(self.memory) < batch_size:
    #         return
    #     try:
    #         minibatch = random.sample(self.memory, batch_size)
    #         for state, action, reward, next_state, done in minibatch:
    #             next_state = np.array(next_state).reshape(1, -1)  # Ensure next_state is a 2D array
    #             target = reward if done else reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
    #             state = np.array(state).reshape(1, -1)  # Ensure state is a 2D array
    #             target_f = self.model.predict(state)
    #             target_f[0][action] = target
    #             self.model.fit(state, target_f, epochs=1, verbose=0, batch_size=1)
    #     except Exception as ex:
    #         print(f"An exception occurred during replay: {ex}")
    #         # Decide on how to handle the exception. For example, you could skip this round of training or take some corrective action.

    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        """Train the model using randomly sampled experiences from the memory."""
        if len(self.memory) < batch_size:
            return
        try:
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                # Here states should already be in the correct shape and type
                target = reward if done else reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
                target_f = self.model.predict(state, verbose=0) #test if verbose removes pringting bars, else remove...
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0, batch_size=1)
        except Exception as ex:
            print(f"An exception occurred during replay: {ex}")
            # Handle the exception appropriately.

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    # def load(self, path_to_weights):
    #     """Load saved model weights."""
    #     try:
    #         self.model.load_weights(path_to_weights)
    #     except Exception as ex:
    #         print(f"An exception occurred loading weights: {ex}")
    #         # Handle the exception, perhaps by initializing the model weights from scratch

    # def save(self, path_to_weights):
    #     """Save model weights."""
    #     try:
    #         self.model.save_weights(path_to_weights)
    #     except Exception as ex:
    #         print(f"An exception occurred saving weights: {ex}")
    #         # Handle the exception, for example, by trying to save to a different location or notifying the user

    def load(self, path_to_model):
        """Load saved model."""
        try:
            self.model = load_model(path_to_model)
            self.update_target_model()  # Ensure target model is also updated
        except Exception as ex:
            print(f"An exception occurred loading the model: {ex}")

    def save(self, path_to_model):
        """Save the complete model."""
        try:
            self.model.save(path_to_model)
        except Exception as ex:
            print(f"An exception occurred saving the model: {ex}")


    def reset(self):
        """Reset the agent state between episodes."""
        self.memory.clear()
        # Reset epsilon to initial value
        # self.epsilon = self.initial_epsilon
        # Optionally reset model weights
        # self.model.set_weights(self.target_model.get_weights())
