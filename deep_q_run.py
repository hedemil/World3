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

actions = [0.9, 0.95, 1.0, 1.05, 1.1]  # Action space
control_signals = ['pptd', 'fioas', 'alai']
num_states = 27
num_actions = len(actions)
num_control_signals = len(control_signals)


# Generate all combinations
action_combinations = list(itertools.product(actions, repeat=len(control_signals)))



def discretize_pop(pop):
    if pop < 5e9: return 0
    elif pop < 7e9: return 1
    else: return 2

def discretize_le(le):
    if le < 50: return 0
    elif le < 70: return 1
    else: return 2

def discretize_fr(fr):
    if fr < 1.5: return 0
    elif fr < 2.5: return 1
    else: return 2



def get_state_vector(pop, le, fr):
    pop_index = discretize_pop(pop)
    le_index = discretize_le(le)
    fr_index = discretize_fr(fr)
    # Return a numpy array with the state represented as a vector
    return np.array([pop_index, le_index, fr_index]).reshape(1, -1)

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
state_size = 3  # For example: population, life expectancy, food ratio
action_size = len(action_combinations)  # Assume 5 possible actions for simplicity
agent = DQNAgent(state_size, action_size)
episodes = 100
batch_size = 32
year_step = 5
year_max = 2200
year_start = 2000

prev_data_optimal, world3_frst = run_world3_simulation(year_min=1900, year_max=2000)


for year in range(year_start, year_max + 1, year_step):
    # Get the current state in vector form
    current_pop = prev_data_optimal['init_vars']['population']['pop'][-1]
    current_le = prev_data_optimal['init_vars']['population']['le'][-1]
    current_fr = prev_data_optimal['init_vars']['agriculture']['fr'][-1]
    state_vector = get_state_vector(current_pop, current_le, current_fr)
    
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
    title="World3 Simulation from 1900 to 2100, optimal policy"
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