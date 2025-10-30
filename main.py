"""
Todo update documentation
"""
import os
import shutil
import numpy as np

from tqdm import tqdm

from utilities.utils import write_positions_to_file, define_mass_lookup_tables, get_particle_mass, get_particle_type, \
    get_atom_type, get_positions_velocities_masses, get_molecule_index, get_atome_charge, get_water_bonds, \
    get_water_angles, generate_simple_water_positions, plot_water_velocities, print_green, print_yellow, print_red
from force_fields.functions import update_positions_and_velocities, update_velocity_using_forces, \
    correct_velocities_based_on_temperature
from utilities.constants import number_particles, dimensions, simulation_steps, simulation_box_size, time_step, \
    simulation_directory, lennard_jones_parameters, mass_dictionary, boltzman_constant, desired_temperatures, \
    water_bond_spring_constant, water_bond_length, water_angle_spring_constant, water_angle, atome_charge_dictionary, \
    coulombs_constant

for desired_temperature in desired_temperatures:
    particle_dictionary = {}
    for particle_index in range(number_particles):
        particle_dictionary[str(particle_index)] = {"position": simulation_box_size * np.random.rand(1, dimensions),
                                                    "velocity": simulation_box_size * np.random.rand(1,
                                                                                                     dimensions) - 0.5,
                                                    "mass": get_particle_mass(particle_index=particle_index,
                                                                              molecule_type="water",
                                                                              mass_dictionary=mass_dictionary),
                                                    "particle_type": get_particle_type(particle_index=particle_index,
                                                                                       molecule_type="water"),
                                                    "atom_type": get_atom_type(particle_index=particle_index,
                                                                               molecule_type="water"),
                                                    "molecule_index": get_molecule_index(particle_index=particle_index,
                                                                                         molecule_type="water"),
                                                    "electrical_charge": get_atome_charge(particle_index=particle_index,
                                                                                          molecule_type="water",
                                                                                          atome_charge_dictionary=atome_charge_dictionary)}

    _, velocities, masses, particle_types, molecule_indexes, electrical_charges, atom_types = get_positions_velocities_masses(
        particle_dictionary=particle_dictionary, number_particles=number_particles)

    positions = generate_simple_water_positions(number_of_water=number_particles // 3,
                                                simulation_box_size=simulation_box_size, initial_hydrogen_offset=1,
                                                max_occupancy=0.7)

    bonds = get_water_bonds(number_particles=number_particles, water_bond_spring_constant=water_bond_spring_constant,
                            bond_length=water_bond_length)
    angles = get_water_angles(number_particles=number_particles,
                              water_angle_spring_constant=water_angle_spring_constant, water_angle=water_angle)

    boundary_conditions = ["reflective"]

    for boundary_condition in boundary_conditions:

        simulation_destination = os.path.join(simulation_directory, f"{boundary_condition}_{desired_temperature}K")

        # Clear the directory before running the simulation
        if os.path.exists(simulation_destination):
            shutil.rmtree(simulation_destination)

        # Create the simulation directory if it does not exist
        os.makedirs(simulation_destination, exist_ok=True)

        for iteration_index in tqdm(range(simulation_steps),
                                    desc=f"Running {boundary_condition}_{desired_temperature}K Simulation :"):
            # Debug the velocities
            print_green("Initial Velocities", add_separators=True)
            plot_water_velocities(velocities=velocities, number_particles=number_particles,
                                  simulation_box_size=simulation_box_size,scale=True)

            # First get velocities based on potential enegies
            # currently only lennard_jones interactions
            velocities, acceleration = update_velocity_using_forces(positions=positions, velocities=velocities,
                                                                    bonds=bonds,
                                                                    sigma=lennard_jones_parameters["sigma"],
                                                                    epsilon=lennard_jones_parameters["epsilon"],
                                                                    angles=angles, molecule_indexes=molecule_indexes,
                                                                    electrical_charges=electrical_charges,
                                                                    coulombs_constant=coulombs_constant,
                                                                    time_step=time_step, masses=masses,
                                                                    atom_types=atom_types, dimensions=dimensions)

            # Debug the velocities
            print_yellow("Velocities After Applied Forces", add_separators=True)
            plot_water_velocities(velocities=velocities, number_particles=number_particles,
                                  simulation_box_size=simulation_box_size,scale=True)

            # Correct the velocities based on temperatures
            velocities = correct_velocities_based_on_temperature(velocities=velocities, masses=masses,
                                                                 boltzman_constant=boltzman_constant,
                                                                 desired_temperature=desired_temperature)
            # Debug the velocities
            print_red("Velocities After Temperature Correction", add_separators=True)
            plot_water_velocities(velocities=velocities, number_particles=number_particles,
                                  simulation_box_size=simulation_box_size,scale=True)

            # Update positions based on velocities
            # Also take into account boundary types in order to manage particles at the simulation boundary
            positions, velocities = update_positions_and_velocities(positions=positions, velocities=velocities,
                                                                    simulation_box_size=simulation_box_size,
                                                                    boundary_conditions=boundary_condition,
                                                                    time_step=time_step)

            write_positions_to_file(positions=positions, simulation_box_size=simulation_box_size,
                                    simulation_directory=simulation_destination, iteration_index=iteration_index,
                                    particle_types=particle_types)
            exit()