"""
Todo update documentation
"""
import os
import numpy as np

from utilities.constants import number_particles, dimensions, simulation_steps, simulation_box_size, time_step, \
    simulation_directory
from utilities.utils import update_positions_and_velocities, round_up_array, write_positions_to_file

# Initialize random positions
positions = 100.0 * np.random.rand(number_particles, dimensions)

# Here we allow negative velocities
# Possibility for them to just point in different directions
velocities = 100.0 * (np.random.rand(number_particles, dimensions) - 0.5)


boundary_conditions = ["periodic", "reflective"]

for boundary_condition in boundary_conditions:

    simulation_destination = os.path.join(simulation_directory, boundary_condition)

    # Create the simulation directory if it does not exist
    os.makedirs(simulation_destination, exist_ok=True)

    for iteration_index in range(1000):
        positions, velocities = update_positions_and_velocities(positions=positions, velocities=velocities,
                                                                simulation_box_size=simulation_box_size,
                                                                boundary_conditions=boundary_condition,
                                                                time_step=time_step)

        write_positions_to_file(positions=positions, simulation_box_size=simulation_box_size,
                                simulation_directory=simulation_destination, iteration_index=iteration_index)
