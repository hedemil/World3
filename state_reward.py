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
            normalized_val = (value - min_val) / range_val

            # Assign the normalized value to a category
            if normalized_val < 0.25:
                category = 0  # Represents the range [0, 0.25)
            elif normalized_val < 0.5:
                category = 1  # Represents the range [0.25, 0.5)
            elif normalized_val < 0.75:
                category = 2  # Represents the range [0.5, 0.75)
            else:
                category = 3  # Represents the range [0.75, 1.0]

            normalized_state[key] = category
        return normalized_state

    

# Reward calculation
def calculate_reward(current_world):
    reward = 0
    # io_derivative = calculate_derivative(current_world.time, current_world.io)
    # io = current_world.io[-1]
    # so_derivative = calculate_derivative(current_world.time, current_world.so)

    # if abs(io_derivative[-1]) < 3e10:
    #     reward += 10
    # elif abs(io_derivative[-1]) < 2e10:
    #     reward += 100
    # else:
    #     reward -= 1000

    # if current_world.pop[-1] < 6e9:
    #     if io < 2e12:
    #         reward += 1
    #     elif io < 1.5e12:
    #         reward += 10
    #     elif io < 1e12:
    #         reward += 100
    # else:
    #     if abs(so_derivative[-1]) < 4e10:
    #         reward += 10
    #     elif abs(so_derivative[-1]) < 2e10:
    #         reward += 100
    #     else:
    #         reward += 0
    #     if current_world.so[-1] > 3e12:
    #         reward += 100
    #     elif current_world.so[-1] > 2e12:
    #         reward += 10
    #     else:
    #         reward -= 1000
    
    le_derivative = calculate_derivative(current_world.time, current_world.le)

    if current_world.le[-1] > 40:
        if abs(le_derivative[-1]) < 0.2:
            reward += 100
        elif abs(le_derivative[-1]) < 0.4:
            reward += 10
    else:
        reward -= 10
    
    


    
    # birth_death = current_world.cbr[-1] / current_world.cdr[-1]
    # if  0.9 <= birth_death <= 1.1:
    #     reward += 10
    # if birth_death < 0.75:
    #     reward -= 1000

    # # if the life expectancy is increasing
    # if current_world.le[-1]/current_world.le[-2] > 1:
    #     reward += 10

    # # World collapses if nrfr goes below 0.5, see Will Thissen
    # reward -= 100000 if current_world.nrfr[-1] < 0.5 else 0 

    # # if healt service allocation and effective health care is increasing
    # if current_world.hsapc[-1]/current_world.hsapc[-2] > 1:
    #     reward += 10
    # if current_world.ehspc[-1]/current_world.ehspc[-2] > 1:
    #     reward += 10

    # # if the mortality rate is decreasing
    # if current_world.m1[-1]/current_world.m1[-2] < 1:
    #     reward += 10
    # if current_world.m1[-1]/current_world.m1[-2] > 1.25:
    #     reward -= 1000

    # if current_world.m2[-1]/current_world.m2[-2] < 1:
    #     reward += 100 
    # if current_world.m2[-1]/current_world.m2[-2] > 1.25:
    #     reward -= 1000
    

    return reward

def calculate_derivative(time, values):
    derivatives = []
    # Use central difference method for the interior points
    for i in range(1, len(time) - 1):
        dy_dx = (values[i + 1] - values[i]) / (time[i + 1] - time[i])
        derivatives.append(dy_dx)

    # Use forward difference for the first point
    dy_dx_first = (values[1] - values[0]) / (time[1] - time[0])
    derivatives.insert(0, dy_dx_first)

    # Use backward difference for the last point
    dy_dx_last = (values[-1] - values[-2]) / (time[-1] - time[-2])
    derivatives.append(dy_dx_last)

    return derivatives