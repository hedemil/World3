import numpy as np
import random
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.initializers import VarianceScaling
from keras.losses import Huber, MeanSquaredError
import os
import tensorflow as tf

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, tau=0.01, epsilon=1.0, epsilon_decay=0.97, epsilon_min=0.01, memory_size=2000, verbose=0, model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
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

    def update_target_model(self):
        """Soft update model parameters."""
        q_model_theta = self.model.get_weights()
        target_model_theta = self.target_model.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_model_theta, target_model_theta):
            target_weight = target_weight * (1 - self.tau) + q_weight * self.tau
            target_model_theta[counter] = target_weight
            counter += 1
        self.target_model.set_weights(target_model_theta)



    def _build_model(self):
        """Neural Net for Deep-Q learning Model."""
        init = VarianceScaling(scale=2, mode='fan_in', distribution='uniform')
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu', kernel_initializer=init))
        model.add(Dropout(0.2)) # Help reduce overfitting
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu')) # Add extra layer
        # model.add(Dropout(0.2))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # def update_target_model(self):
    #     """Copy weights from model to target_model."""
    #     self.target_model.set_weights(self.model.get_weights())

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

    def replay(self, batch_size):
        """Train the model using randomly sampled experiences from the memory."""
        if len(self.memory) < batch_size:
            return  # Ensure there are enough samples in the memory

        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch], dtype=bool)  # Ensure dones is a boolean array

        states = states.reshape((batch_size, -1))
        next_states = next_states.reshape((batch_size, -1))

        # Predict the next state Q-values from the target network for stability
        next_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1)

        # Compute the target Q-values for all actions; only update the action taken
        targets = rewards + (self.gamma * max_next_q_values * (~dones))

        # Get current Q-values predictions for all actions, only adjust those taken
        current_q_values = self.model.predict(states, verbose=0)
        current_q_values[np.arange(batch_size), actions] = targets

        # Train the model on the states and the updated Q-values
        self.model.fit(states, current_q_values, epochs=1, verbose=0, batch_size=batch_size)

        # Soft update the target model every step
        # self.update_target_model()

    def epsilon_dec(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def evaluate_model(self, batch_size):
        """Evaluate the model using a set of experiences from the memory."""
        if len(self.memory) < batch_size:
            print("Not enough samples in memory to evaluate model.")
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        states = states.reshape((batch_size, -1))
        next_states = next_states.reshape((batch_size, -1))
        
        # Predict the Q-values for the current states using the policy network
        current_q_values = self.model.predict(states, verbose=0)

        # Predict the Q-values for the next states using the target network for stability
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Compute the target Q-values
        target_q_values = np.copy(current_q_values)
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Evaluate the model: compare predicted Q-values (current_q_values) with your computed 'target_q_values'
        loss = self.model.evaluate(states, target_q_values, verbose=1)
        print(f"Evaluation loss: {loss}")




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
