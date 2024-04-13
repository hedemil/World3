import numpy as np

# def normalize_state(p1, p2, p3, p4, hsapc, ehspc, time):
#     # Placeholder normalization function - replace with actual logic
#     norm_p1 = p1 / 3e9
#     norm_p2 = p2 / 3e9
#     norm_p3 = p3 / 3e9
#     norm_p4 = p4 / 3e9
#     norm_hsapc = hsapc / 100  # Assuming hsapc max is 100 for normalization
#     norm_ehspc = ehspc / 100  # Assuming ehspc max is 100 for normalization
#     norm_time = (time - 2000) / 200  # Assuming year range 2000-2200 for normalization
#     return np.array([norm_p1, norm_p2, norm_p3, norm_p4, norm_hsapc, norm_ehspc, norm_time]).reshape(1, -1)


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

    def normalize_state(self, state):
        # Normalize each state variable using the updated statistics
        normalized_state = {}
        for key, value in state.items():
            min_val = self.min_values.get(key, 0)
            max_val = self.max_values.get(key, 1)
            range_val = max_val - min_val if max_val > min_val else 1
            normalized_state[key] = (value - min_val) / range_val
        return normalized_state

# Reward calculation
def calculate_reward(current_world):
    reward = 0
    cbr = current_world.cbr[-1]
    cdr = current_world.cdr[-1]
    birth_death = current_world.cbr[-1] / current_world.cdr[-1]
    if  0.9 <= birth_death <= 1.1:
        reward += 10
    if birth_death < 0.75:
        reward -= 1000

    # if the life expectancy is increasing
    le1 = current_world.le[-1]
    le2 = current_world.le[-2]
    if current_world.le[-1]/current_world.le[-2] > 1:
        reward += 10

    # World collapses if nrfr goes below 0.5, see Will Thissen
    nrfr = current_world.nrfr[-1]
    reward -= 100000 if current_world.nrfr[-1] < 0.5 else 0 

    # if healt service allocation and effective health care is increasing
    hsapc1 = current_world.hsapc[-1]
    hsapc2 = current_world.hsapc[-2]
    ehspc1 = current_world.ehspc[-1]
    ehspc2 = current_world.ehspc[-2]
    if current_world.hsapc[-1]/current_world.hsapc[-2] > 1:
        reward += 10
    if current_world.ehspc[-1]/current_world.ehspc[-2] > 1:
        reward += 10

    # if the mortality rate is decreasing
    m11 = current_world.m1[-1]
    m12 = current_world.m1[-2]
    if current_world.m1[-1]/current_world.m1[-2] < 1:
        reward += 10
    if current_world.m1[-1]/current_world.m1[-2] > 1.25:
        reward -= 1000

    m21 = current_world.m2[-1]
    m22 = current_world.m2[-2]
    if current_world.m2[-1]/current_world.m2[-2] < 1:
        reward += 100 
    if current_world.m2[-1]/current_world.m2[-2] > 1.25:
        reward -= 1000
    

    return reward


# Actions and control signals setup
actions = [0.5, 1, 1.5]  # Action space
control_signals = ['icor', 'scor', 'fioac', 'isopc', 'fioas'] 

episodes = 10