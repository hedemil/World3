import matplotlib.pyplot as plt
import numpy as np
import itertools

from pyworld3 import World3, world3
from pyworld3.utils import plot_world_variables

params = {"lines.linewidth": "3"}
plt.rcParams.update(params)

# Parameters
alpha = 0.3  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.2  # Exploration rate
actions = [0.9, 0.95, 1.0, 1.05, 1.1]  # Action space
control_signals = ['lmhs', 'fioas', 'alai']
num_states = 27
num_actions = len(actions)
num_control_signals = len(control_signals)


# Generate all combinations
action_combinations = list(itertools.product(actions, repeat=len(control_signals)))
num_action_combos = len(action_combinations)

# Mapping each combination to an index
action_to_index = {combo: i for i, combo in enumerate(action_combinations)}

# Initialize Q-table
Q = np.zeros((num_states, num_action_combos))

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



def get_state(pop, le, fr):
    pop_index = discretize_pop(pop)
    le_index = discretize_le(le)
    fr_index = discretize_fr(fr)
    num_le_bins = 3
    num_fr_bins = 3
    # Calculate a unique state index
    state_index = pop_index * (num_le_bins * num_fr_bins) + le_index * num_fr_bins + fr_index
    return state_index

# Q-learning update
def update_Q(state, action_index, reward, next_state):
    future = np.max(Q[next_state])
    Q[state, action_index] = Q[state, action_index] + alpha * (reward + gamma * future - Q[state, action_index])

# Reward calculation
def calculate_reward(births, deaths, life_exp, health_service_pc, pop):
    reward = 0
    if births / deaths < 0.9:
        reward += 0
    elif births / deaths <= 1.1:
        reward += 100
    else:
        reward += 0
    reward += 0 if life_exp < 55 else 100
    reward += 0 if health_service_pc < 50 else 100
    reward -= 10000 if pop < 6e9 or pop > 8e9 else 0
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

# Simulation loop
# Initial setup for the World3 model
learning_episodes = 500  
exploraion_episode = 50
initial_year = 2000
final_year = 2100
year_step = 5


for episode in range(learning_episodes):
    # Run the first simulation
    prev_data, world3_start = run_world3_simulation(year_min=1900, year_max=2000)

     # Run the model with actions every 5th year
    for year in range(initial_year, final_year + 1, year_step):
        current_pop = prev_data['init_vars']['population']['pop'][-1]
        current_le = prev_data['init_vars']['population']['le'][-1]
        current_fr = prev_data['init_vars']['agriculture']['fr'][-1]
        state = get_state(current_pop, current_le, current_fr)
        
        if np.random.rand() < epsilon:  # Exploration
            action_combination_index = np.random.choice(len(action_combinations))
        else:  # Exploitation
            state = get_state(current_pop, current_le, current_fr)
            action_combination_index = np.argmax(Q[state])

        # Retrieve the action combination using the selected index
        selected_action_combination = action_combinations[action_combination_index]

        # Now apply these actions to the control signals
        control_variables_actions = list(zip(control_signals, selected_action_combination))
        prev_data['control_signals'] = update_control(control_variables_actions, prev_data['control_signals'])

        prev_data, world3_current = run_world3_simulation(year_min=year, year_max=year + year_step, prev_run_data=prev_data, ordinary_run=False, k_index=prev_data["world_props"]["k"])

        # Calculate reward and update Q-table
        next_pop = world3_current.pop[-1]
        next_le = world3_current.le[-1]
        next_fr = world3_current.fr[-1]
        next_birth = world3_current.cbr[-1]
        next_death = world3_current.cdr[-1]
        
        reward = calculate_reward(next_birth, next_death, next_le, next_fr, next_pop)
        next_state = get_state(next_pop, next_le, next_fr)
        
        update_Q(state, action_combination_index, reward, next_state)

        epsilon *= 0.95

print(Q)
optimal_policy = np.argmax(Q, axis=1)
print("Optimal policy (state -> action index):", optimal_policy)

prev_data_optimal, world3_frst = run_world3_simulation(year_min=1900, year_max=2000)

for year in range(initial_year, final_year + 1, year_step):
    # Get the current state
    current_pop = prev_data_optimal['init_vars']['population']['pop'][-1]
    current_le = prev_data_optimal['init_vars']['population']['le'][-1]
    current_fr = prev_data_optimal['init_vars']['agriculture']['fr'][-1]
    state = get_state(current_pop, current_le, current_fr)
    
    # Use the optimal policy to find the optimal action combination index
    optimal_action_combination_index = optimal_policy[state]
    
    # Retrieve the optimal action combination
    optimal_action_combination = action_combinations[optimal_action_combination_index]
    
    # Construct the list of control signals and their corresponding actions
    control_variables_actions = list(zip(control_signals, optimal_action_combination))
    
    # Update the control signals for the next simulation
    prev_data_optimal['control_signals'] = update_control(control_variables_actions, prev_data_optimal['control_signals'])
    
    # Run the simulation for the next time step using the updated control signals
    prev_data_optimal, world3_optimal = run_world3_simulation(year_min=year, year_max=year + year_step, prev_run_data=prev_data_optimal, ordinary_run=False, k_index=prev_data_optimal["world_props"]["k"])

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
x_pos = 0.05  
y_pos = 0.95  
vertical_offset = 0.05  


ax = plt.gcf().gca()

for var, label in zip(variables, labels):
    max_value = np.max(var)
    ax.text(x_pos, y_pos, f'{label} Max: {max_value:.2f}', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
    y_pos -= vertical_offset  
plt.show()