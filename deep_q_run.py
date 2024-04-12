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
from state_reward import normalize_state


params = {"lines.linewidth": "3"}
plt.rcParams.update(params)

# Actions and control signals setup
actions = [0.8, 1, 1.2]  # Action space
control_signals = ['icor', 'scor', 'fioac', 'isopc', 'fioas', 'nruf', 'fcaor']

# Generate all action combinations
action_combinations = list(itertools.product(actions, repeat=len(control_signals)))


# Define the environment/simulation parameters
state_size = 7  # Number of components in the state vector
action_size = len(action_combinations)
agent = DQNAgent(state_size, action_size)
episodes = 10
batch_size = 32
year_step = 5
year_max = 2200
year_start = 2000

model_path = 'final_model.weights.h5'
agent.load(model_path)



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
        k_index = prev_run_prop['k']
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
        prev_control[control_signal + '_control'] = action_value*prev_control['initial_value'][control_signal + '_control']
    return prev_control


prev_data_optimal, world3_frst = run_world3_simulation(year_min=1900, year_max=2000)


for year in range(year_start, year_max + 1, year_step):
    # Get the current state in vector form
    current_state = normalize_state(prev_data_optimal['init_vars']['population']['p1'][-1],
                                     prev_data_optimal['init_vars']['population']['p2'][-1],
                                     prev_data_optimal['init_vars']['population']['p3'][-1],
                                     prev_data_optimal['init_vars']['population']['p4'][-1],
                                     prev_data_optimal['init_vars']['population']['hsapc'][-1],
                                     prev_data_optimal['init_vars']['population']['ehspc'][-1],
                                     prev_data_optimal['world_props']['time'][-1])
    
    # Use the DQN model to find the optimal action
    action_index = agent.act(current_state)
    
    # Retrieve the optimal action combination based on the DQN model's decision
    optimal_action_combination = action_combinations[action_index]
    
    # Construct the list of control signals and their corresponding actions
    control_variables_actions = list(zip(control_signals, optimal_action_combination))
    
    # Update the control signals for the next simulation
    prev_data_optimal['control_signals'] = update_control(control_variables_actions, prev_data_optimal['control_signals'])
    
    # Run the simulation for the next time step using the updated control signals
    prev_data_optimal, world3_optimal = run_world3_simulation(year_min=year, year_max=year + 5, prev_run_data=prev_data_optimal, ordinary_run=False)

    
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