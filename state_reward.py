import numpy as np

def normalize_state(p1, p2, p3, p4, hsapc, ehspc, time):
    # Placeholder normalization function - replace with actual logic
    norm_p1 = p1 / 3e9
    norm_p2 = p2 / 3e9
    norm_p3 = p3 / 3e9
    norm_p4 = p4 / 3e9
    norm_hsapc = hsapc / 100  # Assuming hsapc max is 100 for normalization
    norm_ehspc = ehspc / 100  # Assuming ehspc max is 100 for normalization
    norm_time = (time - 2000) / 200  # Assuming year range 2000-2200 for normalization
    return np.array([norm_p1, norm_p2, norm_p3, norm_p4, norm_hsapc, norm_ehspc, norm_time]).reshape(1, -1)


# Reward calculation
def calculate_reward(current_world):
    reward = 0
    birth_death = current_world.cbr[-1] / current_world.cdr[-1]
    if  0.9 <= birth_death <= 1.1:
        reward += 100
    else:
        reward += 0
    reward += 0 if current_world.le[-1] < 55 else 100
    reward += 0 if current_world.hsapc[-1] < 50 else 100
    reward -= 10000 if current_world.pop[-1] < 6e9 or current_world.pop[-1] > 8e9 else 0
    reward -= 100000 if current_world.nrfr[-1] < 0.5 else 0 # World collapses if nrfr goes below 0.5, see Will Thissen
    return reward
