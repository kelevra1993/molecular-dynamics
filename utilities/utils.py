"""
# TODO To be documented
"""
import os
import numpy as np

from utilities.constants import simulation_box_size


def update_positions_and_velocities(positions, velocities, time_step, simulation_box_size, boundary_conditions):
    if boundary_conditions not in ["periodic", "reflective"]:
        raise (f"Boundary Conditions Were Not Set For The Simulation Please Set Them To Either :"
               f"'periodic' or 'reflective'")

    # First Naively Update positions
    updated_positions = (positions + (time_step * velocities))

    # Simple Periodic Conditions : Just keep positions in the boundary box
    if boundary_conditions == "periodic":
        updated_positions = updated_positions % simulation_box_size
        updated_velocities = velocities

    if boundary_conditions == "reflective":
        updated_positions, updated_velocities = apply_reflection_to_positions_and_velocities(
            positions=updated_positions, velocities=velocities, simulation_box_size=simulation_box_size)

    return updated_positions, updated_velocities


def apply_reflection_to_positions_and_velocities(positions, velocities, simulation_box_size):
    updated_velocities = velocities
    updated_positions = positions

    number_particles, dimensions = positions.shape

    for particle_index in range(number_particles):
        for axis in range(dimensions):

            # Point back into the box simply by inverting position and velocity
            if positions[particle_index][axis] < 0:
                updated_positions[particle_index][axis] = - positions[particle_index][axis]
                updated_velocities[particle_index][axis] = - velocities[particle_index][axis]

            # Point back into the box by getting symetrical position compared to the axis
            # and invert velocity for the axis
            if positions[particle_index][axis] > simulation_box_size:
                updated_positions[particle_index][axis] = 2 * simulation_box_size - positions[particle_index][axis]
                updated_velocities[particle_index][axis] = - velocities[particle_index][axis]

    return updated_positions, updated_velocities


def write_positions_to_file(positions, simulation_box_size, simulation_directory, iteration_index):
    file_path = os.path.join(simulation_directory, f"{iteration_index}.dump")
    with open(file_path, "w") as output_file:
        output_file.write("ITEM: TIMESTEP\n")
        output_file.write(f"{iteration_index}\n")  # time step
        output_file.write("ITEM: NUMBER OF ATOMS\n")
        output_file.write(f"{len(positions)}\n")  # number of atoms
        output_file.write("ITEM: BOX BOUNDS pp pp pp\n")  # pp = periodic BCs
        output_file.write(f"0 {simulation_box_size}\n")
        output_file.write(f"0 {simulation_box_size}\n")
        output_file.write(f"0 {simulation_box_size}\n")
        output_file.write("ITEM: ATOMS id mol type x y z\n")
        for particle_index in range(len(positions)):
            output_file.write(f"{particle_index} {particle_index} {1} "
                              f"{positions[particle_index][0]} "
                              f"{positions[particle_index][1]} "
                              f"{positions[particle_index][2]}\n")


def round_up_array(numpy_array, decimals=4):
    return np.round(numpy_array, decimals)
