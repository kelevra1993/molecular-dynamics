import os
import json
import numpy as np
from typing import Dict, Any


def get_target_water_velocity_distributions(number_particles: int, mass_dictionary: Dict[str, float],
                                            boltzman_constant: float, desired_temperature: float) -> Dict[str, np.ndarray]:
    """
    Generates target velocity distributions for hydrogen and oxygen atoms.

    This function calculates the target speed norms for hydrogen and oxygen atoms
    based on the Maxwell-Boltzmann distribution for a given temperature. The standard
    deviation (sigma) for the velocity distribution is derived from the temperature
    and particle mass.

    Args:
        number_particles (int): The total number of particles in the simulation.
        mass_dictionary (Dict[str, float]): A dictionary mapping atom types to their masses.
        boltzman_constant (float): The Boltzmann constant.
        desired_temperature (float): The target temperature for the simulation in Kelvin.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing sorted arrays of target speed
                               norms for 'hydrogen' and 'oxygen' atoms.
    """
    hydrogen_indices = [index for index in range(number_particles) if index % 3 != 1]
    oxygen_indices = [index for index in range(number_particles) if index % 3 == 1]

    # Todo document how we get this sigma for better comprehension
    sigma_hydrogen = np.sqrt(boltzman_constant * desired_temperature / mass_dictionary["hydrogen"])
    ideal_hydrogen_velocities = np.random.randn(len(hydrogen_indices), 3) * sigma_hydrogen
    target_hydrogen_speed_norms = np.linalg.norm(ideal_hydrogen_velocities, axis=1)

    # Todo document how we get this sigma for better comprehension
    sigma_oxygen = np.sqrt(boltzman_constant * desired_temperature / mass_dictionary["oxygen"])
    ideal_oxygen_velocities = np.random.randn(len(oxygen_indices), 3) * sigma_oxygen
    target_oxygen_speed_norms = np.linalg.norm(ideal_oxygen_velocities, axis=1)

    # TODO document what is in this dictionary
    # TODO CORRECT FOR THE VELOCITY DIVERGENCE
    # Sorted target norms so that we do not have to do it at each iteration
    sorted_target_hydrogen_speeds_norms = np.sort(target_hydrogen_speed_norms)
    sorted_target_oxygen_speeds_norms = np.sort(target_oxygen_speed_norms)

    target_velocity_distributions = {"hydrogen": sorted_target_hydrogen_speeds_norms,
                                     "oxygen": sorted_target_oxygen_speeds_norms}

    return target_velocity_distributions


def write_positions_to_file(positions: np.ndarray, simulation_box_size: float, simulation_directory: str,
                            iteration_index: int, particle_types: np.ndarray) -> None:
    """
    Writes the atomic positions for a single timestep to a .dump file.

    The output format is compatible with visualization software like OVITO.

    Args:
        positions (np.ndarray): Array of atomic positions.
        simulation_box_size (float): The size of the simulation box.
        simulation_directory (str): The directory where the output file will be saved.
        iteration_index (int): The current simulation step.
        particle_types (np.ndarray): Array of particle types.
    """
    file_path = os.path.join(simulation_directory, f"{iteration_index}.dump")
    with open(file_path, "w") as output_file:
        output_file.write("ITEM: TIMESTEP\n")
        output_file.write(f"{iteration_index}\n")
        output_file.write("ITEM: NUMBER OF ATOMS\n")
        output_file.write(f"{len(positions)}\n")
        output_file.write("ITEM: BOX BOUNDS pp pp pp\n")
        output_file.write(f"0 {simulation_box_size}\n")
        output_file.write(f"0 {simulation_box_size}\n")
        output_file.write(f"0 {simulation_box_size}\n")
        output_file.write("ITEM: ATOMS id mol type x y z\n")
        for particle_index in range(len(positions)):
            output_file.write(f"{particle_index} {particle_index} {particle_types[particle_index]} "
                              f"{positions[particle_index][0]} "
                              f"{positions[particle_index][1]} "
                              f"{positions[particle_index][2]}\n")


def round_up_array(numpy_array: np.ndarray, decimals: int = 4) -> np.ndarray:
    """
    Rounds the elements of a NumPy array to a specified number of decimal places.

    Args:
        numpy_array (np.ndarray): The array to be rounded.
        decimals (int): The number of decimal places to round to.

    Returns:
        np.ndarray: The rounded array.
    """
    return np.round(numpy_array, decimals)


def print_blue(output: str, add_separators: bool = False) -> None:
    """
    Prints the output string in blue color.

    Args:
        output (str): The string to print.
        add_separators (bool): If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[94m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[94m" + "\033[1m" + output + "\033[0m")
        print("\033[94m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[94m" + "\033[1m" + output + "\033[0m")


def print_green(output: str, add_separators: bool = False) -> None:
    """
    Prints the output string in green color.

    Args:
        output (str): The string to print.
        add_separators (bool): If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[32m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[32m" + "\033[1m" + output + "\033[0m")
        print("\033[32m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[32m" + "\033[1m" + output + "\033[0m")


def print_yellow(output: str, add_separators: bool = False) -> None:
    """
    Prints the output string in yellow color.

    Args:
        output (str): The string to print.
        add_separators (bool): If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[93m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[93m" + "\033[1m" + output + "\033[0m")
        print("\033[93m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[93m" + "\033[1m" + output + "\033[0m")


def print_red(output: str, add_separators: bool = False) -> None:
    """
    Prints the output string in red color.

    Args:
        output (str): The string to print.
        add_separators (bool): If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[91m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[91m" + "\033[1m" + output + "\033[0m")
        print("\033[91m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[91m" + "\033[1m" + output + "\033[0m")


def print_bold(output: str, add_separators: bool = False) -> None:
    """
    Prints the output string in bold font.

    Args:
        output (str): The string to print.
        add_separators (bool): If True, prints separators before and after the output.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[1m" + str(length * "-") + "\033[0m")
        print("\033[1m" + output + "\033[0m")
        print("\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[1m" + output + "\033[0m")


def print_dictionary(dictionary: Dict[Any, Any], indent: int = 4) -> None:
    """
    Prints a dictionary in a nicely formatted JSON-like structure.

    Args:
        dictionary (Dict[Any, Any]): The dictionary to print.
        indent (int): The number of spaces to use for indentation.
    """
    print(json.dumps(dictionary, indent=indent))


def read_json(file_path: str) -> dict:
    """
    Reads a JSON file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON file to be read.

    Returns:
        dict: The contents of the JSON file as a Python dictionary.
    """

    with open(file_path) as file:
        return json.load(file)
