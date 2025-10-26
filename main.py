"""
Todo update documentation
"""
import os
import shutil
import numpy as np

from utilities.utils import round_up_array, write_positions_to_file, define_mass_lookup_tables, get_particle_mass, \
    get_positions_velocities_masses
from force_fields.functions import update_positions_and_velocities, update_velocity_using_forces, \
    correct_velocities_based_on_temperature
from utilities.constants import number_particles, dimensions, simulation_steps, simulation_box_size, time_step, \
    simulation_directory, lennard_jones_paramaters, mass_dictionary, boltzman_constant, desired_temperature

particle_dictionary = {}
for particle_index in range(number_particles):
    particle_dictionary[str(particle_index)] = {"position": 100 * np.random.rand(1, dimensions),
                                                "velocity": 100 * np.random.rand(1, dimensions),
                                                "mass": get_particle_mass(particle_index=particle_index,
                                                                          number_particles=number_particles,
                                                                          available_masses=[1, 10]),
                                                "particle_type": 0 if particle_index < number_particles / 2 else 1,

                                                }

positions, velocities, masses, particle_types = get_positions_velocities_masses(particle_dictionary=particle_dictionary,
                                                                                number_particles=number_particles)

# # Initialize random positions
# positions = 100.0 * np.random.rand(number_particles, dimensions)
#
# # Here we allow negative velocities
# # Possibility for them to just point in different directions
# velocities = 100.0 * (np.random.rand(number_particles, dimensions) - 0.5)

# TODO To be removed just testing

# TODO Just for testing will be removed afterwards
beta_testing = False

if beta_testing:
    positions = np.array([[1 + simulation_box_size / 2, simulation_box_size / 2, simulation_box_size / 2],
                          [-1 + simulation_box_size / 2, simulation_box_size / 2, simulation_box_size / 2], ])
    velocities = np.array([[0, 0, 0], [0, 0, 0], ])
    masses, particle_types = define_mass_lookup_tables(number_particles=2, available_masses=[1, 1])

boundary_conditions = ["periodic", "reflective"]

for boundary_condition in boundary_conditions:

    simulation_destination = os.path.join(simulation_directory, boundary_condition)

    # Clear the directory before running the simulation
    if os.path.exists(simulation_destination):
        shutil.rmtree(simulation_destination)

    # Create the simulation directory if it does not exist
    os.makedirs(simulation_destination, exist_ok=True)

    for iteration_index in range(simulation_steps):
        # First get velocities based on potential enegies
        # currently only lennard_jones interactions
        velocities, acceleration = update_velocity_using_forces(positions=positions, velocities=velocities,
                                                                sigma=lennard_jones_paramaters["sigma"],
                                                                epsilon=lennard_jones_paramaters["epsilon"],
                                                                time_step=time_step, mass_dictionary=mass_dictionary)

        # Correct the velocities based on temperatures
        velocities = correct_velocities_based_on_temperature(velocities=velocities, masses=masses,
                                                             boltzman_constant=boltzman_constant,
                                                             desired_temperature=desired_temperature)

        # Update positions based on velocities
        # Also take into account boundary types in order to manage particles at the simulation boundary
        positions, velocities = update_positions_and_velocities(positions=positions, velocities=velocities,
                                                                simulation_box_size=simulation_box_size,
                                                                boundary_conditions=boundary_condition,
                                                                time_step=time_step)

        write_positions_to_file(positions=positions, simulation_box_size=simulation_box_size,
                                simulation_directory=simulation_destination, iteration_index=iteration_index,
                                particle_types=particle_types)
