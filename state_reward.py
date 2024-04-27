import numpy as np


# Try pretraining to define the states before actually running, e.g do a standard run to get min-max. 
class StateNormalizer:
    def __init__(self):
        self.min_values = {}
        self.max_values = {}

    def update_stats(self, state):
        for key, value in state.items():
            if key in self.min_values:
                self.min_values[key] = min(value, self.min_values[key])
                self.max_values[key] = max(value, self.max_values[key])
            else:
                self.min_values[key] = value
                self.max_values[key] = value
    
    def normalize_state(self, state):
        normalized_state = {}
        for key, value in state.items():
            if key not in self.min_values:
                # Handle unseen key error or log a warning
                raise ValueError(f"Key {key} not found in running statistics.")
            min_val = self.min_values[key]
            max_val = self.max_values[key]
            range_val = max_val - min_val if max_val > min_val else 1
            normalized_val = (value - min_val) / range_val
            normalized_state[key] = normalized_val
        return normalized_state


    # Change to let num_bins be a parameter
    # def normalize_state(self, state, num_bins):
    #     normalized_state = {}
    #      # Defines the number of discrete states as 10 intervals [0, 0.1, ..., 0.9, 1.0]
    #     for key, value in state.items():
    #         min_val = self.min_values.get(key, 0)
    #         max_val = self.max_values.get(key, 1)
    #         range_val = max_val - min_val if max_val > min_val else 1
    #         # Normalize to continuous range [0, 1]
    #         normalized_val = (value - min_val) / range_val
    #         # Round to the nearest tenth
    #         discretized_val = round(normalized_val * num_bins) / num_bins
    #         normalized_state[key] = discretized_val
    #     return normalized_state
# Reward calculation
def calculate_reward(current_world):
    reward = 0
    
    le_value = current_world.le[-1]
    le_derivative = calculate_derivative(current_world.le)

    # Adjusted penalty for very low LE values
    if le_value < 20:
        reward -= 500  # Reduced penalty for extremely low LE
    elif le_value < 30:
        reward -= 300  # Smaller penalty for low LE

    # Encourage growth until LE reaches 60
    elif le_value < 60:
        reward += 50 if le_derivative > 0 else -50

    # When LE is above 60, encourage improving and maintaining high LE
    elif le_value >= 60:
        reward += 50
        # Encourage stability in high LE scenarios
        if abs(le_derivative) < 0.1:
            reward += 50  # Reward for stability
        elif abs(le_derivative) < 0.2:
            reward += 25  # Lesser reward for moderate stability
        else:
            reward -= 50 * abs(le_derivative)  # Penalty increases with instability

    # Penalize drastic drops in LE no matter the current value
    if le_derivative < -0.2:
        reward -= 100 * abs(le_derivative)  # Large penalty proportional to the rate of decrease
    
    return reward

def calculate_derivative(values):
    # Simple numerical differentiation: finite difference
    return (values[-1]-values[-2])
    #return np.gradient(values, times)