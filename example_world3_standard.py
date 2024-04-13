# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from pyworld3 import World3
from pyworld3.utils import plot_world_variables

params = {"lines.linewidth": "3"}
plt.rcParams.update(params)

world3 = World3(year_max=2200)
world3.set_world3_control()
world3.init_world3_constants()
world3.init_world3_variables()
world3.set_world3_table_functions()
world3.set_world3_delay_functions()
world3.run_world3(fast=False)

variables = [world3.m1, world3.m2, world3.hsapc, world3.ehspc]
labels = ["M1", "M2", "HSAPC", "EHSPC"]

# Plot the combined results
plot_world_variables(
    world3.time,
    variables,
    labels,
        [[0, 1.5*max(world3.m1)], [0, 1.5*max(world3.m2)], [0, 1.5*max(world3.hsapc)],  [0, 1.5*max(world3.ehspc)]],
    figsize=(10, 7),
    title="World3 Simulation from 1900 to 2200, standard"
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

# plot_world_variables(
#     world3.time,
#     [world3.nrfr, world3.iopc, world3.fpc, world3.pop, world3.ppolx],
#     ["NRFR", "IOPC", "FPC", "POP", "PPOLX"],
#     [[0, 1], [0, 1e3], [0, 1e3], [0, 16e9], [0, 32]],
#     img_background="./img/fig7-7.png",
#     figsize=(7, 5),
#     title="World3 standard run - General",
# )
# plt.savefig("fig_world3_standard_general.pdf")

# plot_world_variables(
#     world3.time,
#     [world3.fcaor, world3.io, world3.tai, world3.aiph, world3.fioaa],
#     ["FCAOR", "IO", "TAI", "AI", "FIOAA"],
#     [[0, 1], [0, 4e12], [0, 4e12], [0, 2e2], [0, 0.201]],
#     img_background="./img/fig7-8.png",
#     figsize=(7, 5),
#     title="World3 standard run - Capital sector",
# )
# plt.savefig("fig_world3_standard_capital.pdf")

# plot_world_variables(
#     world3.time,
#     [world3.ly, world3.al, world3.fpc, world3.lmf, world3.pop],
#     ["LY", "AL", "FPC", "LMF", "POP"],
#     [[0, 4e3], [0, 4e9], [0, 8e2], [0, 1.6], [0, 16e9]],
#     img_background="./img/fig7-9.png",
#     figsize=(7, 5),
#     title="World3 standard run - Agriculture sector",
# )
# plt.savefig("fig_world3_standard_agriculture.pdf")
