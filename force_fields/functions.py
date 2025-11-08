import numpy as np
import math
from typing import List, Dict, Tuple

# Numpy Options
np.set_printoptions(linewidth=int(1e5))


######################################
# Computation Of Gradient Potentials #
######################################
def compute_lennard_jones_gradient_potential(positions: np.ndarray, molecule_indexes: np.ndarray,
                                             atom_types: np.ndarray, atom_index: int, sigma: Dict,
                                             epsilon: Dict) -> np.ndarray:
    """
    Calculates the gradient of the Lennard-Jones potential for a single atom.

    Args:
        positions (np.ndarray): Array of all particle positions.
        molecule_indexes (np.ndarray): Array of molecule indices for each particle.
        atom_types (np.ndarray): Array of atom types for each particle.
        atom_index (int): The index of the atom for which to calculate the potential.
        sigma (dict): Dictionary of sigma values for Lennard-Jones potential.
        epsilon (dict): Dictionary of epsilon values for Lennard-Jones potential.

    Returns:
        np.ndarray: The gradient of the Lennard-Jones potential for the specified atom.
    """
    number_particles = positions.shape[0]

    atom_type = atom_types[atom_index]

    # Create sigma and epsilon pairwise values
    sigma_vector = []
    for other_atom_index in range(number_particles):
        other_atom_type = atom_types[other_atom_index]

        # hydrogen-hydrogen or oxygen-oxygen
        if atom_type == other_atom_type:
            sigma_vector.append(sigma[f"{atom_type}_{atom_type}"])
        else:
            sigma_vector.append(sigma["oxygen_hydrogen"])

    epsilon_vector = []
    for other_atom_index in range(number_particles):
        other_atom_type = atom_types[other_atom_index]

        # hydrogen-hydrogen or oxygen-oxygen
        if atom_type == other_atom_type:
            epsilon_vector.append(epsilon[f"{atom_type}_{atom_type}"])
        else:
            epsilon_vector.append(epsilon["oxygen_hydrogen"])

    sigma_vector = np.array(sigma_vector)
    epsilon_vector = np.array(epsilon_vector)

    # For same molecule set epsilon and sigma to zero
    molecule_index = molecule_indexes[atom_index]
    for other_atom_index in range(number_particles):
        other_molecule_index = molecule_indexes[other_atom_index]
        if molecule_index == other_molecule_index:
            epsilon_vector[other_atom_index] = 0
            sigma_vector[other_atom_index] = 0

    # Remove item-wise sigma and epsilon
    sigma_vector = np.delete(sigma_vector, atom_index)
    epsilon_vector = np.delete(epsilon_vector, atom_index)

    vectors_to_atom_i = positions[atom_index] - positions

    # Remove the atom index, since pairwise interaction with itself does not matter
    vectors_to_atom_i = np.delete(vectors_to_atom_i, atom_index, axis=0)

    # Get distance to atom i
    absolute_distances_to_atom_i = np.linalg.norm(vectors_to_atom_i, axis=1)

    # Get Potential Gradient
    # gradient of attractive term
    attractive_part = epsilon_vector * (sigma_vector ** 6) / (absolute_distances_to_atom_i ** 8)

    # gradient of repulsive term
    repulsive_part = epsilon_vector * (2 * (sigma_vector ** 12)) / (absolute_distances_to_atom_i ** 14)

    gradient_term = repulsive_part - attractive_part

    # Becareful here * is not a matrix multiplication in the common sense.
    gradient_term_applied_to_vector = np.transpose(np.transpose(vectors_to_atom_i) * gradient_term)

    # Sum up everything so that we get the gradient in the x, y and z direction
    lennard_jones_gradient_potential = -24 * (np.sum(gradient_term_applied_to_vector, axis=0))

    return lennard_jones_gradient_potential


def compute_bond_energy_gradient_potential(positions: np.ndarray, bonds: List, dimensions: int) -> np.ndarray:
    """
    Computes the gradient of the bond energy potential for all atoms.

    Args:
        positions (np.ndarray): Array of all particle positions.
        bonds (list): List of bonds, where each bond is [atom1_idx, atom2_idx, length, strength].
        dimensions (int): The number of dimensions in the simulation.

    Returns:
        np.ndarray: An array of bond gradient potentials for each atom.
    """
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


def compute_angle_gradient_potential(positions: np.ndarray, angles: List, dimensions: int) -> np.ndarray:
    """
    Computes the gradient of the angle potential for all atoms.

    Args:
        positions (np.ndarray): Array of all particle positions.
        angles (list): List of angles, where each angle is [atom1_idx, atom2_idx, atom3_idx, angle, strength].
        dimensions (int): The number of dimensions in the simulation.

    Returns:
        np.ndarray: An array of angle gradient potentials for each atom.
    """
    number_particles = positions.shape[0]

    angle_gradient_potentials = np.zeros([number_particles, dimensions])

    for atom_index in range(number_particles):
        for angle_index in range(len(angles)):
            first_atom = angles[angle_index][0]
            second_atom = angles[angle_index][1]
            third_atom = angles[angle_index][2]

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
                    angle_gradient_potential = (
                            negative_angle_potential_derivative_with_respect_to_theta * numerator / denominator)
                    angle_gradient_potentials[atom_index] += angle_gradient_potential

                # Dealing with middle atom
                if second_atom == atom_index:
                    first_part = -(first_vector + second_vector)
                    second_part = dot_product * first_vector / (first_vector_norm ** 2)
                    third_part = dot_product * second_vector / (second_vector_norm ** 2)
                    numerator = (first_part - second_part + third_part) / (first_vector_norm * second_vector_norm)
                    angle_gradient_potential = negative_angle_potential_derivative_with_respect_to_theta * numerator / denominator
                    angle_gradient_potentials[atom_index] += angle_gradient_potential

    return angle_gradient_potentials


def compute_coulomb_force(positions: np.ndarray, charges: np.ndarray, particle_index: int,
                          molecule_indexes: np.ndarray, coulombs_constant: float) -> np.ndarray:
    """
    Calculates the electrostatic (Coulomb) force on a single particle.

    Args:
        positions (np.ndarray): Array of all particle positions.
        charges (np.ndarray): Array of charges for each particle.
        particle_index (int): The index of the particle for which to calculate the force.
        molecule_indexes (np.ndarray): Array of molecule indices for each particle.
        coulombs_constant (float): The Coulomb's constant.

    Returns:
        np.ndarray: The electrostatic force vector on the specified particle.
    """
    number_particles = positions.shape[0]

    particle_charge = charges[particle_index]

    # Set particles of the same molecule to have 0.0 chare relative to each other
    # No coulomb force between them
    standardised_particle_charges_per_molecule = charges.copy()

    for other_particle in range(number_particles):
        if molecule_indexes[particle_index] == molecule_indexes[other_particle]:
            standardised_particle_charges_per_molecule[other_particle] = 0.0

    # Remove the particle_index from the standardised particle charges per molecule
    standardised_particle_charges_per_molecule = np.delete(standardised_particle_charges_per_molecule, particle_index)

    # Compute vectors and remove 0 vector to avoid dividing by 0
    vectors = positions[particle_index] - positions
    vectors = np.delete(vectors, particle_index, axis=0)

    vector_norms = np.linalg.norm(vectors, axis=1)

    partial_force_value_vector = particle_charge * standardised_particle_charges_per_molecule * coulombs_constant * (
            (1.0 / vector_norms) ** 3)

    # Sum up all forces that apply to the charge
    electrostatic_force = np.sum(np.transpose(np.transpose(vectors) * partial_force_value_vector), axis=0)

    return electrostatic_force


################################################
# End Of Computation Of Gradient of Potentials #
################################################

def update_velocity_using_forces(positions: np.ndarray, velocities: np.ndarray, bonds: List, angles: List,
                                 molecule_indexes: np.ndarray, electrical_charges: np.ndarray,
                                 coulombs_constant: float, time_step: float, sigma: Dict, epsilon: Dict,
                                 masses: np.ndarray, atom_types: np.ndarray, dimensions: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Updates particle velocities based on the calculated forces.

    This function calculates the total force on each particle from Lennard-Jones,
    bond, angle, and Coulomb interactions, then computes the acceleration and
    updates the velocities.

    Args:
        positions (np.ndarray): Array of all particle positions.
        velocities (np.ndarray): Array of all particle velocities.
        bonds (list): List of all bonds.
        angles (list): List of all angles.
        molecule_indexes (np.ndarray): Array of molecule indices for each particle.
        electrical_charges (np.ndarray): Array of electrical charges for each particle.
        coulombs_constant (float): The Coulomb's constant.
        time_step (float): The simulation time step.
        sigma (dict): Dictionary of sigma values for Lennard-Jones potential.
        epsilon (dict): Dictionary of epsilon values for Lennard-Jones potential.
        masses (np.ndarray): Array of masses for each particle.
        atom_types (np.ndarray): Array of atom types for each particle.
        dimensions (int): The number of dimensions in the simulation.

    Returns:
        tuple: A tuple containing the updated velocities and the calculated accelerations.
    """
    number_particles = positions.shape[0]
    # Compute Forces interacting on all molecules based on leonard jones interactions
    forces = -np.array([compute_lennard_jones_gradient_potential(positions=positions, molecule_indexes=molecule_indexes,
                                                                 atom_types=atom_types, atom_index=index, sigma=sigma,
                                                                 epsilon=epsilon) for index in range(len(positions))])

    # Add forces emmanating from bond potentials
    bond_gradient_potentials = compute_bond_energy_gradient_potential(positions=positions, bonds=bonds,
                                                                      dimensions=dimensions)
    forces -= bond_gradient_potentials

    # Add forces emmanating from bond angle potentials
    bond_angle_potentials = compute_angle_gradient_potential(positions=positions, angles=angles, dimensions=dimensions)

    forces -= bond_angle_potentials

    # Add electrostatic forces
    coulumb_forces = [
        compute_coulomb_force(positions=positions, charges=electrical_charges, particle_index=particle_index,
                              molecule_indexes=molecule_indexes, coulombs_constant=coulombs_constant) for particle_index
        in range(number_particles)]

    forces += coulumb_forces

    accelerations = np.transpose(np.transpose(forces) / masses)

    # Integrate since we consider that the acceleration is constant
    updated_velocities = velocities + (accelerations * time_step)

    return updated_velocities, accelerations


def correct_velocities_based_on_temperature(velocities: np.ndarray, masses: np.ndarray, boltzman_constant: float,
                                            desired_temperature: float) -> np.ndarray:
    """
    Applies a velocity scaling thermostat to adjust particle velocities.

    This function scales the velocities of all particles to match a desired temperature.

    Args:
        velocities (np.ndarray): Array of all particle velocities.
        masses (np.ndarray): Array of masses for each particle.
        boltzman_constant (float): The Boltzmann constant.
        desired_temperature (float): The target temperature.

    Returns:
        np.ndarray: The corrected velocities.
    """
    number_particles = len(velocities)
    kinetic_energy = 0.5 * sum(sum(masses * np.transpose(velocities * velocities)))
    average_kinetic_energy = kinetic_energy / number_particles

    # Get current temperature of the system and move it back to desired temperature
    current_temperature = (2 / 3) * average_kinetic_energy / boltzman_constant

    # Correction value
    correction_value = np.sqrt(desired_temperature / current_temperature)

    corrected_velocities = correction_value * velocities

    return corrected_velocities


def correct_velocities_based_on_target_velocity_distributions(velocities: np.ndarray,
                                                              target_velocity_distributions: Dict) -> np.ndarray:
    """
    Adjusts velocities to match a target Maxwell-Boltzmann distribution.

    Args:
        velocities (np.ndarray): Array of all particle velocities.
        target_velocity_distributions (dict): Dictionary with target speed norms for 'hydrogen' and 'oxygen'.

    Returns:
        np.ndarray: The corrected velocities.
    """
    number_particles = len(velocities)
    hydrogen_indices = [index for index in range(number_particles) if index % 3 != 1]
    oxygen_indices = [index for index in range(number_particles) if index % 3 == 1]

    hydrogen_velocities = velocities[hydrogen_indices]
    oxygen_velocities = velocities[oxygen_indices]

    # Get particle speed and normalize them
    current_hydrogen_speed_norms = np.linalg.norm(hydrogen_velocities, axis=1)
    current_hydrogen_directions = hydrogen_velocities / current_hydrogen_speed_norms.reshape(-1, 1)

    current_oxygen_speed_norms = np.linalg.norm(oxygen_velocities, axis=1)
    current_oxygen_directions = oxygen_velocities / current_oxygen_speed_norms.reshape(-1, 1)

    target_hydrogen_speeds_norms = target_velocity_distributions["hydrogen"]
    target_oxygen_speeds_norms = target_velocity_distributions["oxygen"]

    # Get ranks of target particle velocities (they had already been sorted)
    current_hydrogen_ranks = np.argsort(target_hydrogen_speeds_norms)
    current_oxygen_ranks = np.argsort(target_oxygen_speeds_norms)

    # Equalize by mapping distributions
    new_hydrogen_speed_norms = np.zeros(len(hydrogen_indices))
    for target_index, hydrogen_index in enumerate(current_hydrogen_ranks):
        new_hydrogen_speed_norms[hydrogen_index] = target_hydrogen_speeds_norms[target_index]

    new_oxygen_speed_norms = np.zeros(len(oxygen_indices))
    for target_index, oxygen_index in enumerate(current_oxygen_ranks):
        new_oxygen_speed_norms[oxygen_index] = target_oxygen_speeds_norms[target_index]

    corrected_hydrogen_velocities = current_hydrogen_directions * new_hydrogen_speed_norms.reshape(-1, 1)
    corrected_oxygen_velocities = current_oxygen_directions * new_oxygen_speed_norms.reshape(-1, 1)

    corrected_velocities = merge_velocities(hydrogen_velocities=corrected_hydrogen_velocities,
                                            oxygen_velocities=corrected_oxygen_velocities,
                                            number_particles=number_particles)

    return corrected_velocities


def merge_velocities(hydrogen_velocities: np.ndarray, oxygen_velocities: np.ndarray,
                     number_particles: int) -> np.ndarray:
    """
    Merges separate velocity arrays for hydrogen and oxygen atoms into a single array.

    Args:
        hydrogen_velocities (np.ndarray): Array of hydrogen velocities.
        oxygen_velocities (np.ndarray): Array of oxygen velocities.
        number_particles (int): The total number of particles.

    Returns:
        np.ndarray: A single array containing all particle velocities.
    """
    hydrogen_indices = [index for index in range(number_particles) if index % 3 != 1]
    oxygen_indices = [index for index in range(number_particles) if index % 3 == 1]
    velocities = np.zeros((number_particles, 3))

    for index, hydrogen_index in enumerate(hydrogen_indices):
        velocities[hydrogen_index] = hydrogen_velocities[index]

    for index, oxygen_index in enumerate(oxygen_indices):
        velocities[oxygen_index] = oxygen_velocities[index]

    return velocities


def compute_temperature(velocities: np.ndarray, mass_dictionary: Dict, message_description: str,
                        boltzman_constant: float):
    """
    Calculates and prints the current temperature of the system.

    Args:
        velocities (np.ndarray): Array of all particle velocities.
        mass_dictionary (dict): Dictionary of masses for each atom type.
        message_description (str): A message to print before the temperature.
        boltzman_constant (float): The Boltzmann constant.
    """
    number_particles = len(velocities)

    hydrogen_indices = [index for index in range(number_particles) if index % 3 != 1]
    oxygen_indices = [index for index in range(number_particles) if index % 3 == 1]

    hydrogen_velocities = velocities[hydrogen_indices]
    oxygen_velocities = velocities[oxygen_indices]

    kinetic_hydrogen_energy = 0.5 * sum(
        sum(mass_dictionary["hydrogen"] * np.transpose(hydrogen_velocities * hydrogen_velocities)))
    kinetic_oxygen_energy = 0.5 * sum(
        sum(mass_dictionary["oxygen"] * np.transpose(oxygen_velocities * oxygen_velocities)))
    average_kinetic_energy = (kinetic_hydrogen_energy + kinetic_oxygen_energy) / number_particles

    # Get current temperature of the system and move it back to desired temperature
    current_temperature = (2 / 3) * average_kinetic_energy / boltzman_constant
    print(f"{message_description} : {current_temperature}")


def update_positions_and_velocities(positions: np.ndarray, velocities: np.ndarray, time_step: float,
                                    simulation_box_size: float, boundary_conditions: str) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Updates particle positions using the Verlet algorithm and applies boundary conditions.

    Args:
        positions (np.ndarray): Array of all particle positions.
        velocities (np.ndarray): Array of all particle velocities.
        time_step (float): The simulation time step.
        simulation_box_size (float): The size of the simulation box.
        boundary_conditions (str): The type of boundary conditions ('periodic' or 'reflective').

    Returns:
        tuple: A tuple containing the updated positions and velocities.
    """
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


def apply_reflection_to_positions_and_velocities(positions: np.ndarray, velocities: np.ndarray,
                                                 simulation_box_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies reflective boundary conditions to particles.

    If a particle has moved outside the simulation box, its position is reflected
    back inside, and its velocity component perpendicular to the boundary is inverted.

    Args:
        positions (np.ndarray): Array of all particle positions.
        velocities (np.ndarray): Array of all particle velocities.
        simulation_box_size (float): The size of the simulation box.

    Returns:
        tuple: A tuple containing the updated positions and velocities.
    """
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
