"""
# TODO To be documented
"""
import numpy as np
import math

from utilities.utils import print_blue, print_yellow

# Numpy Options
np.set_printoptions(linewidth=int(1e5))


#############################
# Computation Of Potentials #
#############################
# TODO Remove pairwise potentials for the same molecule so add bond argument
def compute_pairwise_lennard_jones_potentials(positions, bonds, atom_index, sigma, epsilon):
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


# TODO Remove pairwise potentials for the same molecule so add bond argument
def compute_lennard_jones_gradient_potential(positions, bonds, atom_index, sigma, epsilon):
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


def compute_bond_energy_potentials(positions, bonds):
    # bonds is a list that contains these elements
    # element_1=[first_atom_index, second_atom_index, bond length , bond strength]
    # element_2=[first_atom_index, second_atom_index, bond length , bond strength]
    # ...
    number_particles = positions.shape[0]

    # We initialise with 0 bond potentials for every position since each atom is not necessarily bonded
    bond_potentials = np.zeros(number_particles)

    for atom_index in range(number_particles):
        for bond_index in range(len(bonds)):
            first_bonded_atom = bonds[bond_index][0]
            second_bonded_atom = bonds[bond_index][1]

            # Case where our atom has a bond
            if atom_index == first_bonded_atom or atom_index == second_bonded_atom:

                # Find atom bonded to atom_index
                if first_bonded_atom == atom_index:
                    bonded_atom = int(second_bonded_atom)
                else:
                    bonded_atom = int(first_bonded_atom)

                optimal_bond_length = bonds[bond_index][2]
                bond_strength = bonds[bond_index][3]

                vector_bonded_particles = positions[atom_index] - positions[bonded_atom]
                distance_bonded_particles = np.sqrt(sum(vector_bonded_particles * vector_bonded_particles))

                bond_potential = bond_strength * ((distance_bonded_particles - optimal_bond_length) ** 2)

                # Avoid counting it twice since it should also appear on the bonded atom index
                bond_potentials[
                    atom_index] += 0.5 * bond_potential  # # TODO To Be Removed : Just the debugging of bonds  # print(f"Atom Index : {atom_index}  >> Bonded Atom Index : {bonded_atom}  >> Potential {0.5 * bond_potential}")
    return bond_potentials


def compute_bond_energy_gradient_potential(positions, bonds, dimensions):
    # bonds is a list that contains these elements
    # element_1=[first_atom_index, second_atom_index, bond length , bond strength]
    # element_2=[first_atom_index, second_atom_index, bond length , bond strength]
    # ...
    number_particles = positions.shape[0]

    # We initialise with 0's bond gradient potentials for every position since each atom is not necessarily bonded
    bond_gradient_potentials = np.zeros([number_particles, dimensions])

    for atom_index in range(number_particles):
        for bond_index in range(len(bonds)):
            first_bonded_atom = bonds[bond_index][0]
            second_bonded_atom = bonds[bond_index][1]

            # Case where our atom has a bond
            if atom_index == first_bonded_atom or atom_index == second_bonded_atom:

                # Find atom bonded to atom_index
                if first_bonded_atom == atom_index:
                    bonded_atom = int(second_bonded_atom)
                else:
                    bonded_atom = int(first_bonded_atom)

                optimal_bond_length = bonds[bond_index][2]
                bond_strength = bonds[bond_index][3]

                vector_bonded_particles = positions[atom_index] - positions[bonded_atom]
                distance_bonded_particles = np.sqrt(sum(vector_bonded_particles * vector_bonded_particles))

                bond_gradient_potential = (2 * (bond_strength / distance_bonded_particles) * (
                        distance_bonded_particles - optimal_bond_length)) * vector_bonded_particles

                # Add bond gradient potential
                bond_gradient_potentials[atom_index] += bond_gradient_potential

    return bond_gradient_potentials


def compute_angle_potentials(positions):
    # TODO To Be Implemented
    return 0


def compute_angle_gradient_potential(positions, angles, dimensions):
    number_particles = positions.shape[0]

    angle_gradient_potentials = np.zeros([number_particles, dimensions])

    for atom_index in range(number_particles):
        for angle_index in range(len(angles)):
            first_atom = angles[angle_index][0]
            second_atom = angles[angle_index][1]
            third_atom = angles[angle_index][2]

            # TODO If statement to be checked
            if atom_index in [first_atom, second_atom, third_atom]:
                equilibrium_angle = angles[angle_index][3]
                spring_angle_constant = angles[angle_index][4]

                if atom_index in [first_atom, second_atom]:
                    first_vector = positions[first_atom] - positions[second_atom]
                    second_vector = positions[third_atom] - positions[second_atom]
                else:
                    first_vector = positions[third_atom] - positions[second_atom]
                    second_vector = positions[first_atom] - positions[second_atom]

                first_vector_norm = np.sqrt(sum(first_vector * first_vector))
                second_vector_norm = np.sqrt(sum(second_vector * second_vector))

                # Done in order to get the cosine and then the angle
                dot_product = sum(first_vector * second_vector)

                angle_cosine = dot_product / (first_vector_norm * second_vector_norm)

                current_angle = math.acos(angle_cosine)

                negative_angle_potential_derivative_with_respect_to_theta = -2 * spring_angle_constant * (
                        current_angle - equilibrium_angle)

                denominator = np.sqrt(1.0 - (angle_cosine ** 2))

                # Case of terminal atoms in the angle definition
                if first_atom == atom_index or third_atom == atom_index:
                    numerator = (second_vector / (first_vector_norm * second_vector_norm)) - (
                            dot_product / (2 * (first_vector_norm ** 3) * second_vector_norm))
                    angle_potential = (
                            negative_angle_potential_derivative_with_respect_to_theta * numerator / denominator)
                    angle_gradient_potentials[atom_index] += angle_potential
                # Dealing with middle atom
                if second_atom == atom_index:
                    first_part = -(second_vector + first_vector)
                    second_part = dot_product * first_vector / (first_vector_norm ** 2)
                    third_part = dot_product * second_vector / (second_vector_norm ** 2)
                    numerator = (first_part - second_part + third_part)

    print(*angle_gradient_potentials, sep="\n")
    return 0


####################################
# End Of Computation Of Potentials #
####################################

def update_velocity_using_forces(positions, velocities, bonds, time_step, sigma, epsilon, masses, dimensions):
    # Compute Forces interacting on all molecules based on leonard jones interactions
    forces = -np.array(
        [compute_lennard_jones_gradient_potential(positions=positions, atom_index=index, sigma=sigma, epsilon=epsilon)
         for index in range(len(positions))])

    # Add forces emmanating from bond potentials
    bond_gradient_potentials = compute_bond_energy_gradient_potential(positions=positions, bonds=bonds,
                                                                      dimensions=dimensions)
    forces -= bond_gradient_potentials

    accelerations = np.transpose(np.transpose(forces) / masses)

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

    # # Playing around with temperature manipulation
    # # Controlled Velocity Update -- Could be tried
    # alpha = 0.2
    # corrected_velocities = alpha * (correction_value * velocities) + (1 - alpha) * velocities
    # print(100*'-')
    # print(f"Current Temperature : {current_temperature}")
    # average_corrected_kinetic_energy = 0.5 * sum(sum(masses * np.transpose(corrected_velocities * corrected_velocities)))
    # print(f"Corrected Temperature : {(2 / 3) * average_corrected_kinetic_energy / boltzman_constant}")
    # print(100 * '-')

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
