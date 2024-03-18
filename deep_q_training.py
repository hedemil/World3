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

def simulate_step(year, prev_data, action_combination_index, control_signals):
    """
    Simulate one step of the World3 model based on the given action and update control signals.

    :param year: Current year of simulation.
    :param prev_data: Previous run data of the World3 model.
    :param action_combination_index: Index of the selected action combination.
    :param control_signals: List of control signals to be adjusted.
    :return: Tuple of (next_state, reward, done)
    """
    
    # Retrieve the action combination using the selected index
    selected_action_combination = action_combinations[action_combination_index]
    
    # Update control signals based on the selected action
    control_variables_actions = list(zip(control_signals, selected_action_combination))
    prev_data['control_signals'] = update_control(control_variables_actions, prev_data['control_signals'])
    
    # Run the World3 model for the next step
    next_year = year + year_step
    prev_data, world3_current = run_world3_simulation(year_min=year, year_max=next_year, prev_run_data=prev_data, ordinary_run=False)
    
    # Extract necessary variables for state and reward calculation
    current_pop = world3_current.pop[-1]
    current_le = world3_current.le[-1]
    current_fr = world3_current.fr[-1]  # Assuming 'fr' is a food ratio or similar
    
    # Calculate next state
    next_state = get_state_vector(current_pop, current_le, current_fr)
    
    # Calculate reward (this function needs to be defined based on your criteria)
    reward = calculate_reward(world3_current)
    
    # Check if simulation is done (e.g., reached final year)
    done = next_year >= year_max
    
    return next_state, reward, done

# Define the environment / simulation parameters
state_size = 3  # For example: population, life expectancy, food ratio
action_size = len(action_combinations)  # Assume 5 possible actions for simplicity
agent = DQNAgent(state_size, action_size)
episodes = 10
batch_size = 32
year_step = 5
year_max = 2200
year_start = 2000


# Loop over episodes
for e in range(episodes):
    # Run the first simulation
    prev_data, world3_start = run_world3_simulation(year_min=1900, year_max=2000)
    current_pop = prev_data['init_vars']['population']['pop'][-1]
    current_le = prev_data['init_vars']['population']['le'][-1]
    current_fr = prev_data['init_vars']['agriculture']['fr'][-1]
    state = get_state_vector(current_pop, current_le, current_fr)
    for year in (year_start, year_max + 1, year_step):  # Assume a maximum of 50 timesteps per episode
        action = agent.act(state)
        next_state, reward, done = simulate_step(year, prev_data, action, control_signals)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
        
agent.save("your_model.weights.h5")
