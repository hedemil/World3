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
        reward += 10
    if birth_death < 0.75:
        reward -= 1000

    # if the life expectancy is increasing
    if current_world.le[-1]/current_world.le[-2] > 1:
        reward += 10

    # World collapses if nrfr goes below 0.5, see Will Thissen
    reward -= 100000 if current_world.nrfr[-1] < 0.5 else 0 

    # if healt service allocation and effective health care is increasing
    if current_world.hsapc[-1]/current_world.hsapc[-2] > 1:
        reward += 10
    if current_world.ehspc[-1]/current_world.ehspc[-2] > 1:
        reward += 10

    # if the mortality rate is decreasing
    if current_world.m1[-1]/current_world.m1[-2] < 1:
        reward += 10
    if current_world.m1[-1]/current_world.m1[-2] > 1.25:
        reward -= 1000

    if current_world.m2[-1]/current_world.m2[-2] < 1:
        reward += 100 
    if current_world.m2[-1]/current_world.m2[-2] > 1.25:
        reward -= 1000
    

    return reward


# Actions and control signals setup
actions = [0.5, 1, 1.5]  # Action space
control_signals = ['icor', 'scor', 'fioac', 'isopc', 'fioas'] 

episodes = 10