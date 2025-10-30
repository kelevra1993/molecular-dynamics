"""
# TODO To be documented
"""
import os
import json
import numpy as np
import plotext as plt


# todo to be removed
def define_mass_lookup_tables(number_particles, available_masses):
    # Masses in Dalton
    # Particle types
    particle_types = np.zeros(number_particles, dtype=int)
    for index in range(number_particles):
        if index < number_particles / 2:
            particle_types[index] = 0
        else:
            particle_types[index] = 1

    # todo to be changed and documented otherwise
    mass_lookup_table = np.array([available_masses[particle_types[index]] for index in range(number_particles)])

    return mass_lookup_table, particle_types


def get_particle_mass(particle_index, molecule_type, mass_dictionary):
    # Masses in Dalton
    if molecule_type != "water":
        exit("The only covered molecule type is 'water'")

    if molecule_type == "water":
        if particle_index % 3 == 1:
            mass_particle = mass_dictionary["oxygen"]
        else:
            mass_particle = mass_dictionary["hydrogen"]

    return mass_particle


def get_particle_type(particle_index, molecule_type):
    if molecule_type != "water":
        exit("The only covered molecule type is 'water'")

    if molecule_type == "water":
        if particle_index % 3 == 1:
            particle_type = 1
        else:
            particle_type = 0

    return particle_type


def get_atom_type(particle_index, molecule_type):
    if molecule_type != "water":
        exit("The only covered molecule type is 'water'")

    if molecule_type == "water":
        if particle_index % 3 == 1:
            atom_type = "oxygen"
        else:
            atom_type = "hydrogen"

    return atom_type


def get_molecule_index(particle_index, molecule_type):
    if molecule_type != "water":
        exit("The only covered molecule type is 'water'")

    if molecule_type == "water":
        molecule_index = particle_index // 3
    # print(f"Particle Index >> {particle_index} And Molecule Index {molecule_index}")

    return molecule_index


def get_atome_charge(particle_index, molecule_type, atome_charge_dictionary):
    if molecule_type != "water":
        exit("The only covered molecule type is 'water'")

    if molecule_type == "water":
        atome_charges = atome_charge_dictionary[molecule_type]
        if particle_index % 3 == 1:
            atome_charge = atome_charges["oxygen"]
        else:
            atome_charge = atome_charges["hydrogen"]

    return atome_charge


def get_positions_velocities_masses(particle_dictionary, number_particles):
    # Assuming particle_dictionary already exists
    positions = [particle_dictionary[str(i)]["position"] for i in range(number_particles)]
    velocities = [particle_dictionary[str(i)]["velocity"] for i in range(number_particles)]
    masses = [particle_dictionary[str(i)]["mass"] for i in range(number_particles)]
    particle_types = [particle_dictionary[str(i)]["particle_type"] for i in range(number_particles)]
    molecule_indexes = [particle_dictionary[str(i)]["molecule_index"] for i in range(number_particles)]
    electrical_charges = [particle_dictionary[str(i)]["electrical_charge"] for i in range(number_particles)]
    atom_types = [particle_dictionary[str(i)]["atom_type"] for i in range(number_particles)]

    # Optionally, convert to numpy arrays
    positions = np.vstack(positions)  # shape: (number_particles, dimensions)
    velocities = np.vstack(velocities)  # shape: (number_particles, dimensions)
    masses = np.array(masses)  # shape: (number_particles,)
    particle_types = np.array(particle_types)  # shape: (number_particles,)
    molecule_indexes = np.array(molecule_indexes)  # shape: (number_particles,)
    electrical_charges = np.array(electrical_charges)  # shape: (number_particles,)
    atom_types = np.array(atom_types)  # shape: (number_particles,)

    return positions, velocities, masses, particle_types, molecule_indexes, electrical_charges, atom_types


def generate_simple_water_positions(number_of_water, simulation_box_size, initial_hydrogen_offset, max_occupancy):
    target_box_size = int(max_occupancy * simulation_box_size)
    center_factor = (1 - max_occupancy) / 2
    translator_vector = center_factor * simulation_box_size * np.ones(3)

    box_dimensions = np.array([target_box_size] * 3)

    # --- Pre-allocate the final positions array ---
    total_atoms = number_of_water * 3
    all_positions = np.zeros((total_atoms, 3))

    # --- Loop to generate each molecule ---
    for i in range(number_of_water):
        # --- a. Generate a random position for the Oxygen ---
        # This is a random [x, y, z] coordinate inside the box.
        oxygen_position = np.random.rand(3) * box_dimensions

        # --- b. Calculate positions for the two Hydrogens ---
        # Apply small offsets on x and y coordinates.
        hydrogen1_position = oxygen_position + np.array([initial_hydrogen_offset, 0.0, 0.0])
        hydrogen2_position = oxygen_position + np.array([0.0, initial_hydrogen_offset, 0.0])

        # --- c. Add to the main array in [H, O, H] order ---
        # Add correction factor to move them to the center of the simulation box

        start_index = i * 3
        all_positions[start_index] = hydrogen1_position + translator_vector
        all_positions[start_index + 1] = oxygen_position + translator_vector
        all_positions[start_index + 2] = hydrogen2_position + translator_vector

    return all_positions


import numpy as np


def scale_to_0_100(vector):
    vector = np.array(vector, dtype=float)
    v_min, v_max = vector.min(), vector.max()
    if v_max == v_min:
        return np.zeros_like(vector)  # avoid division by zero if all values are equal
    return 100 * (vector - v_min) / (v_max - v_min)


def plot_water_velocities(velocities, number_particles, simulation_box_size, scale):
    velocity_norms = np.linalg.norm(velocities, axis=1)
    if scale:
        velocity_norms = scale_to_0_100(vector=velocity_norms)
        simulation_box_size = 20

    oxygen_velocities = [velocity_norms[v] for v in range(number_particles) if v % 3 == 1]
    hydrogen_velocities = [velocity_norms[v] for v in range(number_particles) if v % 3 != 1]

    bins = np.linspace(velocity_norms.min(), velocity_norms.max(), simulation_box_size // 2)
    bins = np.round(bins,0)

    x = bins[:-1]

    oxygen_counts, _ = np.histogram(oxygen_velocities, bins=bins)
    hyrdogen_counts, _ = np.histogram(hydrogen_velocities, bins=bins)

    plt.bar(x - 0.5, oxygen_counts, width=0.1, label="Oxygen")
    plt.bar(x + 0.5, hyrdogen_counts, width=0.1, label="Hydrogen")
    # plt.hist(oxygen_velocities,bins=simulation_box_size)
    # plt.hist(hydrogen_velocities,bins=simulation_box_size)
    plt.plotsize(width=500)

    plt.show()
    plt.clf()


def get_water_bonds(number_particles, water_bond_spring_constant, bond_length):
    bonds = []
    for i in range(int(number_particles / 3)):
        bonds.append([3 * i, 3 * i + 1, bond_length,
                      water_bond_spring_constant])  # [first_atom_index, second_atom_index, bond length , bond strength]
        bonds.append([3 * i + 1, 3 * i + 2, bond_length, water_bond_spring_constant])

    return bonds


def get_water_angles(number_particles, water_angle_spring_constant, water_angle):
    # [first_atom_index, second_atom_index, third atom index, angle , angle spring constant]
    angles = []
    for i in range(int(number_particles / 3)):
        angles.append([3 * i, 3 * i + 1, 3 * i + 2, water_angle, water_angle_spring_constant])

    return angles


def write_positions_to_file(positions, simulation_box_size, simulation_directory, iteration_index, particle_types):
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
            output_file.write(f"{particle_index} {particle_index} {particle_types[particle_index]} "
                              f"{positions[particle_index][0]} "
                              f"{positions[particle_index][1]} "
                              f"{positions[particle_index][2]}\n")


def round_up_array(numpy_array, decimals=4):
    return np.round(numpy_array, decimals)


def print_blue(output, add_separators=False):
    """
    Prints the output string in blue color.
    :param output: The string that we wish to print in a certain color.
    :param add_separators: If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[94m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[94m" + "\033[1m" + output + "\033[0m")
        print("\033[94m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[94m" + "\033[1m" + output + "\033[0m")


def print_green(output, add_separators=False):
    """
    Prints the output string in green color.
    :param output: The string that we wish to print in a certain color.
    :param add_separators: If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[32m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[32m" + "\033[1m" + output + "\033[0m")
        print("\033[32m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[32m" + "\033[1m" + output + "\033[0m")


def print_yellow(output, add_separators=False):
    """
    Prints the output string in yellow color.
    :param output: The string that we wish to print in a certain color.
    :param add_separators: If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[93m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[93m" + "\033[1m" + output + "\033[0m")
        print("\033[93m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[93m" + "\033[1m" + output + "\033[0m")


def print_red(output, add_separators=False):
    """
    Prints the output string in red color.
    :param output: The string that we wish to print in a certain color.
    :param add_separators: If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[91m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[91m" + "\033[1m" + output + "\033[0m")
        print("\033[91m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[91m" + "\033[1m" + output + "\033[0m")


def print_bold(output, add_separators=False):
    """
    Prints the output string in bold font.
    :param output: The string that we wish to print in bold font.
    :param add_separators: If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[1m" + str(length * "-") + "\033[0m")
        print("\033[1m" + output + "\033[0m")
        print("\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[1m" + output + "\033[0m")


def print_dictionary(dictionary, indent=4):
    dictionary = make_json_serializable(dictionary)
    print(json.dumps(dictionary, indent=indent))


def make_json_serializable(obj):
    """Recursively convert NumPy types to JSON-compatible types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # np.float64, np.int32, etc.
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj
