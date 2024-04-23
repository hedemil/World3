import numpy as np


# Try pretraining to define the states before actually running, e.g do a standard run to get min-max. 
class StateNormalizer:
    def __init__(self):
        # Initialize dictionaries to store the running statistics
        self.min_values = {}
        self.max_values = {}

    def update_stats(self, state):
        # Update the statistics for each state variable
        for key, value in state.items():
            if key in self.min_values:
                self.min_values[key] = min(value, self.min_values[key])
                self.max_values[key] = max(value, self.max_values[key])
            else:
                self.min_values[key] = value
                self.max_values[key] = value
    # Change to let num_bins be a parameter
    def normalize_state(self, state, num_bins):
        normalized_state = {}
         # Defines the number of discrete states as 10 intervals [0, 0.1, ..., 0.9, 1.0]
        for key, value in state.items():
            min_val = self.min_values.get(key, 0)
            max_val = self.max_values.get(key, 1)
            range_val = max_val - min_val if max_val > min_val else 1
            # Normalize to continuous range [0, 1]
            normalized_val = (value - min_val) / range_val
            # Round to the nearest tenth
            discretized_val = round(normalized_val * num_bins) / num_bins
            normalized_state[key] = discretized_val
        return normalized_state


# Reward calculation
def calculate_reward(current_world):
    reward = 0
    
    le_value = current_world.le[-1]
    le_derivative = calculate_derivative(current_world.le)
    
    if le_value < 20:
        reward -= 100

    elif le_value < 30:
        reward -= 50

    # Encourage growth until LE reaches 60
    elif le_value < 60:
        reward += 10 * le_derivative  # Proportional reward based on the rate of increase

    # When LE is above 60, encourage maintaining it and penalize large changes
    elif le_value >= 60:
        reward += 50
        if abs(le_derivative) < 0.1:
            reward += 50  # Big reward for stability
        elif abs(le_derivative) < 0.2:
            reward += 10  # Smaller reward for less stability
        else:
            reward -= 100 * abs(le_derivative)  # Proportional penalty for instability

    # Penalize drastic drops in LE no matter the current value
    if le_derivative < -0.2:
        reward -= 100 * abs(le_derivative)  # Large penalty proportional to the rate of decrease
    
    return reward

def calculate_derivative(values):
    # Simple numerical differentiation: finite difference
    return (values[-1]-values[-5])/5
    #return np.gradient(values, times)