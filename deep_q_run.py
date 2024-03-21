import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from dqn import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import itertools

from pyworld3 import World3, world3
from pyworld3.utils import plot_world_variables


params = {"lines.linewidth": "3"}
plt.rcParams.update(params)

actions = [0.9, 1.0, 1.1]  # Action space
control_signals = ['alai', 'lyf']

""" control_signals = ['alai', 'lyf', 'ifpc', 'lymap', 'llmy', 'fioaa', 
                   'icor', 'scor', 'alic', 'alsc', 'fioac', 'isopc', 
                   'fioas', 'ppgf', 'pptd', 'nruf', 'fcaor'] """
num_states = 81920
num_actions = len(actions)
num_control_signals = len(control_signals)


# Generate all combinations
action_combinations = list(itertools.product(actions, repeat=len(control_signals)))

def discretize_year(time):
    return (time - 2000)//10 + 1


# States for optimizing hsapc
def discretize_p1(p1):
    if p1 < 1e9: return 0
    elif p1 < 2e9: return 1
    elif p1 < 3e9: return 2
    else: return 3

def discretize_p2(p2):
    if p2 < 1e9: return 0
    elif p2 < 2e9: return 1
    elif p2 < 3e9: return 2
    else: return 3

def discretize_p3(p3):
    if p3 < 1e9: return 0
    elif p3 < 2e9: return 1
    elif p3 < 3e9: return 2
    else: return 3

def discretize_p4(p4):
    if p4 < 1e9: return 0
    elif p4 < 2e9: return 1
    elif p4 < 3e9: return 2
    else: return 3

def discretize_hsapc(hsapc):
    if hsapc < 25: return 0
    elif hsapc < 50: return 1
    elif hsapc < 75: return 2
    else: return 3

def discretize_ehspc(ehspc):
    if ehspc < 20: return 0
    elif ehspc < 40: return 1
    elif ehspc < 60: return 2
    else: return 3


def get_state_vector(p1, p2, p3, p4, hsapc, ehspc, time):
    p1_index = discretize_p1(p1)
    p2_index = discretize_p2(p2)
    p3_index = discretize_p3(p3)
    p4_index = discretize_p4(p4)

    hsapc_index = discretize_hsapc(hsapc)
    ehspc_index = discretize_ehspc(ehspc)

    time_index = discretize_year(time)
    # Return a numpy array with the state represented as a vector
    # return np.array([p1_index, hsapc_index, ehspc_index]).reshape(1, -1)

    return np.array([p1_index, p2_index, p3_index, p4_index, hsapc_index, ehspc_index, time_index]).reshape(1, -1)

# Reward calculation
def calculate_reward(current_world):
    reward = 0
    if current_world.cbr[-1] / current_world.cdr[-1] < 0.9:
        reward += 0
    elif current_world.cbr[-1] / current_world.cdr[-1] <= 1.1:
        reward += 100
    else:
        reward += 0
    reward += 0 if current_world.le[-1] < 55 else 100
    reward += 0 if current_world.hsapc[-1] < 50 else 100
    reward -= 10000 if current_world.pop[-1] < 6e9 or current_world.pop[-1] > 8e9 else 0
    return reward

def run_world3_simulation(year_min, year_max, dt=1, prev_run_data=None, ordinary_run=True, k_index=1):
    
    prev_run_prop = prev_run_data["world_props"] if prev_run_data else None

    world3 = World3(
            year_max=year_max,
            year_min=year_min, 
            dt=dt,
            prev_world_prop=prev_run_prop,
            ordinary_run=ordinary_run
        )
    
    if prev_run_data:
        world3.set_world3_control(prev_run_data['control_signals'])
        world3.init_world3_constants()
        world3.init_world3_variables(prev_run_data["init_vars"])
        world3.set_world3_table_functions()
        world3.set_world3_delay_functions(prev_delay=prev_run_data["delay_funcs"])
    else:
        world3.set_world3_control()
        world3.init_world3_constants()
        world3.init_world3_variables()
        world3.set_world3_table_functions()
        world3.set_world3_delay_functions()

    world3.run_world3(fast=False, k_index=k_index)
    state = world3.get_state()
    return state, world3

def update_control(control_signals_actions, prev_control):
    """
    Update control signals based on actions.
    :param control_signals_actions: List of tuples (control_signal, action_value)
    :param prev_control: Previous control signals dictionary
    :return: Updated control signals dictionary
    """
    for control_signal, action_value in control_signals_actions:
        prev_control[control_signal + '_control'] *= action_value
    return prev_control

# Define the environment / simulation parameters
state_size = 7  # For example: population, life expectancy, food ratio
action_size = len(action_combinations)  # Assume 5 possible actions for simplicity
agent = DQNAgent(state_size, action_size)
# Load previously saved model weights
agent.load("final_model.weights.h5")
year_step = 5
year_max = 2200
year_start = 2000

prev_data_optimal, world3_frst = run_world3_simulation(year_min=1900, year_max=2000)


for year in range(year_start, year_max + 1, year_step):
    # Get the current state in vector form
    current_p1 = prev_data_optimal['init_vars']['population']['p1'][-1]
    current_p2 = prev_data_optimal['init_vars']['population']['p2'][-1]
    current_p3 = prev_data_optimal['init_vars']['population']['p3'][-1]
    current_p4 = prev_data_optimal['init_vars']['population']['p4'][-1]
    current_hsapc = prev_data_optimal['init_vars']['population']['hsapc'][-1]
    current_ehspc = prev_data_optimal['init_vars']['population']['ehspc'][-1]
    current_time = prev_data_optimal['world_props']['time'][-1]
    state_vector = get_state_vector(current_p1, current_p2, current_p3, current_p4, current_hsapc, current_ehspc, current_time)
    
    # Use the DQN model to find the optimal action
    action_index = agent.act(state_vector)
    
    # Retrieve the optimal action combination based on the DQN model's decision
    optimal_action_combination = action_combinations[action_index]
    
    # Construct the list of control signals and their corresponding actions
    control_variables_actions = list(zip(control_signals, optimal_action_combination))
    
    # Update the control signals for the next simulation
    prev_data_optimal['control_signals'] = update_control(control_variables_actions, prev_data_optimal['control_signals'])
    
    # Run the simulation for the next time step using the updated control signals
    prev_data_optimal, world3_optimal = run_world3_simulation(year_min=year, year_max=year + 5, prev_run_data=prev_data_optimal, ordinary_run=False, k_index=prev_data_optimal["world_props"]["k"])

variables = [world3_optimal.le, world3_optimal.fr, world3_optimal.sc, world3_optimal.pop]
labels = ["LE", "FR", "SC", "POP"]

# Plot the combined results
plot_world_variables(
    world3_optimal.time,
    variables,
    labels,
        [[0, 100], [0, 4], [0, 6e12],  [0, 10e9]],
    figsize=(10, 7),
    title="World3 Simulation from 1900 to 2200, optimal policy"
)

# Initialize a position for the first annotation
x_pos = 0.05  # Adjust as needed
y_pos = 0.95  # Start from the top, adjust as needed
vertical_offset = 0.05  # Adjust the space between lines

# Use plt.gcf() to get the current figure and then get the current axes with gca()
ax = plt.gcf().gca()

for var, label in zip(variables, labels):
    max_value = np.max(var)
    # Place text annotation within the plot, using figure's coordinate system
    ax.text(x_pos, y_pos, f'{label} Max: {max_value:.2f}', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
    y_pos -= vertical_offset  # Move up for the next line
plt.show()