"""
Todo update documentation
"""
import os
import shutil
import numpy as np

from utilities.utils import round_up_array, write_positions_to_file, define_mass_lookup_tables, get_particle_mass, \
    get_particle_type, get_positions_velocities_masses, print_dictionary, get_molecule_index, get_atome_charge
from force_fields.functions import update_positions_and_velocities, update_velocity_using_forces, \
    correct_velocities_based_on_temperature, compute_coulomb_force
from utilities.constants import number_particles, dimensions, simulation_steps, simulation_box_size, time_step, \
    simulation_directory, lennard_jones_paramaters, mass_dictionary, boltzman_constant, desired_temperature, \
    water_bond_spring_constant, water_bond_length, water_angle_spring_constant, water_angle, atome_charge_dictionary,coulombs_constant

particle_dictionary = {}
for particle_index in range(number_particles):
    particle_dictionary[str(particle_index)] = {"position": simulation_box_size * np.random.rand(1, dimensions),
                                                "velocity": np.random.rand(1, dimensions) - 0.5,
                                                "mass": get_particle_mass(particle_index=particle_index,
                                                                          molecule_type="water",
                                                                          mass_dictionary=mass_dictionary),
                                                "particle_type": get_particle_type(particle_index=particle_index,
                                                                                   molecule_type="water"),
                                                "molecule_index": get_molecule_index(particle_index=particle_index,
                                                                                     molecule_type="water"),
                                                "electrical_charge": get_atome_charge(particle_index=particle_index,
                                                                                      molecule_type="water",
                                                                                      atome_charge_dictionary=atome_charge_dictionary)}

positions, velocities, masses, particle_types, molecule_indexes, electrical_charges = get_positions_velocities_masses(
    particle_dictionary=particle_dictionary, number_particles=number_particles)


def get_water_bonds(number_particles, water_bond_spring_constant, bond_length):
    bonds = []
    for i in range(int(number_particles / 3)):
        bonds.append([3 * i, 3 * i + 1, bond_length,
                      water_bond_spring_constant])  # [first_atom_index, second_atom_index, bond length , bond strength]
        bonds.append([3 * i + 1, 3 * i + 2, bond_length, water_bond_spring_constant])

    return bonds


def get_dihydrogen_bonds(number_particles):
    bonds = []
    for i in range(int(number_particles / 2)):
        bonds.append(
            [2 * i, 2 * i + 1, 1.0, 100.0])  # [first_atom_index, second_atom_index, bond length , bond strength]

    return bonds


def get_water_angles(number_particles, water_angle_spring_constant, water_angle):
    # [first_atom_index, second_atom_index, third atom index, angle , angle spring constant]
    angles = []
    for i in range(int(number_particles / 3)):
        angles.append([3 * i, 3 * i + 1, 3 * i + 2, water_angle, water_angle_spring_constant])

    return angles


bonds = get_water_bonds(number_particles=number_particles, water_bond_spring_constant=water_bond_spring_constant,
                        bond_length=water_bond_length)
angles = get_water_angles(number_particles=number_particles, water_angle_spring_constant=water_angle_spring_constant,
                          water_angle=water_angle)

columb_forces = [compute_coulomb_force(positions=positions, charges=electrical_charges, particle_index=particle_index,
                                       molecule_indexes=molecule_indexes,coulombs_constant=coulombs_constant) for particle_index in range(number_particles)]
exit()
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

boundary_conditions = ["reflective"]

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
        velocities, acceleration = update_velocity_using_forces(positions=positions, velocities=velocities, bonds=bonds,
                                                                sigma=lennard_jones_paramaters["sigma"],
                                                                epsilon=lennard_jones_paramaters["epsilon"],
                                                                time_step=time_step, masses=masses,
                                                                dimensions=dimensions)

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
