"""
# TODO To be documented
"""
import numpy as np

from utilities.utils import print_blue, print_yellow

# Numpy Options
np.set_printoptions(linewidth=int(1e5))


def compute_pairwise_lennard_jones_potentials(positions, atom_index, sigma, epsilon):
    vectors_to_atom_i = positions[atom_index] - positions

    # Remove the atom index, since pairwise interaction with itself does not matter
    vectors_to_atom_i = np.delete(vectors_to_atom_i, atom_index, axis=0)

    # Get distance to atom i
    absolute_distances_to_atom_i = np.linalg.norm(vectors_to_atom_i, axis=1)

    # Get Potential
    # attractive to the power 6
    attractive_part = (sigma / absolute_distances_to_atom_i) ** 6

    # repulsive to the power 12
    repulsive_part = (sigma / absolute_distances_to_atom_i) ** 12

    # Put the lennard jones potential together
    # Here we add up all the potentials between atom i and all other atoms
    lennard_jones_potential = 4 * epsilon * sum(repulsive_part - attractive_part)

    return lennard_jones_potential


def compute_lennard_jones_gradient_potential(positions, atom_index, sigma, epsilon):
    vectors_to_atom_i = positions[atom_index] - positions

    # Remove the atom index, since pairwise interaction with itself does not matter
    vectors_to_atom_i = np.delete(vectors_to_atom_i, atom_index, axis=0)

    # Get distance to atom i
    absolute_distances_to_atom_i = np.linalg.norm(vectors_to_atom_i, axis=1)

    # Get Potential Gradient
    # gradient of attractive term
    attractive_part = (sigma ** 6) / (absolute_distances_to_atom_i ** 8)

    # gradient of repulsive term
    repulsive_part = (2 * (sigma ** 12)) / (absolute_distances_to_atom_i ** 14)

    gradient_term = (repulsive_part - attractive_part)

    # Becareful here * is not a matrix multiplication in the common sense.
    gradient_term_applied_to_vector = np.transpose(np.transpose(vectors_to_atom_i) * gradient_term)

    # Sum up everything so that we get the gradient in the x, y and z direction
    lennard_jones_gradient_potential = -24 * epsilon * (np.sum(gradient_term_applied_to_vector, axis=0))

    return lennard_jones_gradient_potential


def update_velocity_using_forces(positions, velocities, time_step, sigma, epsilon, mass_dictionary):
    # Compute Forces interacting on all molecules based on leonard jones interactions
    forces = -np.array(
        [compute_lennard_jones_gradient_potential(positions=positions, atom_index=index, sigma=sigma, epsilon=epsilon)
         for index in range(len(positions))])

    # TODO Will have to be changed to be better especially how we define our atoms
    accelerations = forces / mass_dictionary["hydrogen"]

    # Integrate since we consider that the acceleration is constant
    updated_velocities = velocities + (accelerations * time_step)

    return updated_velocities, accelerations


def correct_velocities_based_on_temperature(velocities, masses, boltzman_constant, desired_temperature):
    number_particles = len(velocities)
    kinetic_energy = 0.5 * sum(sum(masses * np.transpose(velocities * velocities)))
    average_kinetic_energy = kinetic_energy / number_particles

    # Get current temperature of the system and move it back to desired temperature
    current_temperature = (2 / 3) * average_kinetic_energy / boltzman_constant

    # Correction value
    correction_value = np.sqrt(desired_temperature / current_temperature)

    # # Controlled Velocity Update -- Could be tried
    # alpha = 0.5
    # alpha_corrected_velocities = alpha * (correction_value * velocities) + (1 - alpha) * velocities

    corrected_velocities = correction_value * velocities

    return corrected_velocities


def update_positions_and_velocities(positions, velocities, time_step, simulation_box_size, boundary_conditions):
    if boundary_conditions not in ["periodic", "reflective"]:
        raise (f"Boundary Conditions Were Not Set For The Simulation Please Set Them To Either :"
               f"'periodic' or 'reflective'")

    # First Naively Update positions
    updated_positions = (positions + (time_step * velocities))
    updated_velocities = velocities

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
