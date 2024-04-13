import os
import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time

from dqn import DQNAgent
from state_reward import StateNormalizer, calculate_reward #,normalize_state
from pyworld3 import World3

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Actions and control signals setup
actions = [0.8, 1, 1.2]  # Action space
control_signals = ['icor', 'scor', 'fioac', 'isopc', 'fioas', 'nruf', 'fcaor'] # Add nruf, and fcaor next run.


# Generate all action combinations
action_combinations = list(itertools.product(actions, repeat=len(control_signals)))

# Define the environment/simulation parameters
state_size = 7  # Number of components in the state vector
action_size = len(action_combinations)
agent = DQNAgent(state_size, action_size)
episodes = 100
batch_size = 32
year_step = 5
year_max = 2200
year_start = 2000

# Create an instance of the StateNormalizer
state_normalizer = StateNormalizer()


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
    try:
        prev_data, world3_current = run_world3_simulation(year_min=year, year_max=next_year, prev_run_data=prev_data, ordinary_run=False)
    except Exception as ex:
        print(f"Failed to initialize the World3 simulation year: {year}, exception: {ex}")

    # Extract necessary variables for state and reward calculation
    # current_p1 = world3_current.p1[-1]
    # current_p2 = world3_current.p2[-1]
    # current_p3 = world3_current.p3[-1]
    # current_p4 = world3_current.p4[-1]
    # current_hsapc = world3_current.hsapc[-1]
    # current_ehspc = world3_current.ehspc[-1]
    # current_time = world3_current.time[-1]  
    
    raw_state = {
        'p1': world3_current.p1[-1],
        'p2': world3_current.p2[-1],
        'p3': world3_current.p3[-1],
        'p4': world3_current.p4[-1],
        'hsapc': world3_current.hsapc[-1],
        'ehspc': world3_current.ehspc[-1],
        'time': world3_current.time[-1],
    }

    # Calculate next state
    state_normalizer.update_stats(state=raw_state)
    
    # Normalize the state with the updated statistics
    normalized_state = state_normalizer.normalize_state(state=raw_state)
    next_state = np.array(list(normalized_state.values())).reshape(1, -1)
    
    # Calculate next state
    # next_state = normalize_state(current_p1, current_p2, current_p3, current_p4, current_hsapc, current_ehspc, current_time)
    
    # Calculate reward (this function needs to be defined based on your criteria)
    reward = calculate_reward(world3_current)
    
    # Check if simulation is done (e.g., reached final year)
    done = next_year >= year_max

    
    return prev_data, next_state, reward, done

# Paths for saving models
save_path = "/content/drive/My Drive/Colab Notebooks/"

# Initial state
# current_state = normalize_state(prev_data['init_vars']['population']['p1'][-1],
#                                 prev_data['init_vars']['population']['p2'][-1],
#                                 prev_data['init_vars']['population']['p3'][-1],
#                                 prev_data['init_vars']['population']['p4'][-1],
#                                 prev_data['init_vars']['population']['hsapc'][-1],
#                                 prev_data['init_vars']['population']['ehspc'][-1],
#                                 prev_data['world_props']['time'][-1])

# Create a "state-space" from a normal run
prev_data_state, world3_frst = run_world3_simulation(year_min=1900, year_max=2000)

for year in range(year_start, year_max + 1, year_step):
    # Initial state normalization
    raw_state = {
        'p1': prev_data_state['init_vars']['population']['p1'][-1],
        'p2': prev_data_state['init_vars']['population']['p2'][-1],
        'p3': prev_data_state['init_vars']['population']['p3'][-1],
        'p4': prev_data_state['init_vars']['population']['p4'][-1],
        'hsapc': prev_data_state['init_vars']['population']['hsapc'][-1],
        'ehspc': prev_data_state['init_vars']['population']['ehspc'][-1],
        'time': prev_data_state['world_props']['time'][-1],
    }

    state_normalizer.update_stats(state=raw_state)
    # Run the simulation for the next time step using the updated control signals
    prev_data_state, world3_state = run_world3_simulation(year_min=year, year_max=year + 5, prev_run_data=prev_data_state, ordinary_run=False)

try:
    episode_durations = []
    for e in range(episodes):
        agent.reset()
        start_time = time.time()  # Start timing the episode
        # Initialize start values for simulation
        try:
            prev_data, world3_start = run_world3_simulation(year_min=1900, year_max=2000)
        except Exception as ex:
            print(f"Failed to initialize the World3 simulation: {ex}")

        # Initial state normalization
        raw_state = {
            'p1': prev_data['init_vars']['population']['p1'][-1],
            'p2': prev_data['init_vars']['population']['p2'][-1],
            'p3': prev_data['init_vars']['population']['p3'][-1],
            'p4': prev_data['init_vars']['population']['p4'][-1],
            'hsapc': prev_data['init_vars']['population']['hsapc'][-1],
            'ehspc': prev_data['init_vars']['population']['ehspc'][-1],
            'time': prev_data['world_props']['time'][-1],
        }
        state_normalizer.update_stats(state=raw_state)
        normalized_state = state_normalizer.normalize_state(state=raw_state)
        current_state = np.array(list(normalized_state.values())).reshape(1, -1)
  
        for year in range(year_start, year_max + 1, year_step):
            
            try:
                action_index = agent.act(current_state)
                prev_data, next_state, reward, done = simulate_step(year, prev_data, action_index, control_signals)
                agent.remember(current_state, action_index, reward, next_state, done)
            except ValueError as ve:
                print(f"Model prediction or memory operation failed: {ve}")
            except Exception as ex:
                print(f"An error occurred during the simulation step {e}: {ex}") # Got error after 70 ish episodes...?
                continue  # Depending on the case, you might want to skip or break
            current_state = next_state
            
            if done:
                break

            if len(agent.memory) > batch_size:
                try:
                    agent.replay(batch_size)
                except RuntimeError as re:
                    print(f"Error during training: {re}")
                except Exception as ex:
                    print(f"Unexpected error during training: {ex}")


        # Update the target network at the end of each episode
        agent.update_target_model()

        end_time = time.time()  # End timing the episode
        duration = end_time - start_time
        episode_durations.append(duration)
        print(f"Episode {e+1} completed in {duration:.2f} seconds.")

        if (e + 1) % 100 == 0:
            print(f"Episode: {e + 1}/{episodes}")
            #agent.save(f"{save_path}model_weights_episode_{e+1}.h5")
            try:
                #agent.save(f"{save_path}model_weights_episode_{e+1}.h5")
                agent.save(f"episode_{e+1}_model.keras")
                print(f"Episode: {e + 1}/{episodes} saved sucesfully")
            except Exception as ex:
                print('Failed to save ' f'Episode: {e + 1}/{episodes}, exception: {ex}')

except Exception as ex:
    print(f"An unexpected error occurred: {ex}")

try:
    agent.save('final_model.keras')
    #agent.save("final_model.weights.h5")
    print('Model saved succesfully')

except Exception as ex:
    print('Failed to save model')

# try:
#     #agent.model.save(f"{save_path}final_model.weights.h5")
#     agent.model.save("final_model.h5")  # Saves the full model
#     print('Model saved successfully')
# except Exception as ex:
#     print('Failed to save model:', ex)

# print all episode durations
print("Episode durations:", episode_durations)

