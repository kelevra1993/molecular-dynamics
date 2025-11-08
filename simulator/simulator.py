import os
import shutil
import numpy as np
from tqdm import tqdm

from molecules.water import WaterMolecule

from utilities.utils import read_json, get_target_water_velocity_distributions, write_positions_to_file

from force_fields.functions import update_velocity_using_forces, \
    correct_velocities_based_on_target_velocity_distributions, correct_velocities_based_on_temperature, \
    update_positions_and_velocities

class Simulator:
    def __init__(self, number_water_molecules, simulation_configuration, desired_temperature, boundary_condition):
        """
        Initializes the Simulator object.

        Args:
            number_water_molecules (int): The number of water molecules to include in the simulation.
            simulation_configuration (dict): A dictionary containing the configuration parameters for the simulation.
            desired_temperature (int): The desired temperature for the simulation in Kelvin.
            boundary_condition (str): The boundary condition to use for the simulation (e.g., 'periodic', 'reflective').
        """
        self.number_water_molecules = number_water_molecules
        self.simulation_configuration = simulation_configuration
        self.simulation_box_size = self.simulation_configuration["simulation_box_size"]
        self.time_step = self.simulation_configuration["time_step"]
        self.simulation_steps = self.simulation_configuration["simulation_steps"]
        self.lennard_jones_parameters = self.simulation_configuration["lennard_jones_parameters"]
        self.mass_dictionary = self.simulation_configuration["mass_configurations"]
        self.boltzman_constant = self.simulation_configuration["boltzman_constant"]
        self.desired_temperature = desired_temperature
        self.coulombs_constant = self.simulation_configuration["coulombs_constant"]
        self.dimensions = 3
        self.boundary_condition = boundary_condition

        self._create_simulation_directory()

        (self.positions, self.velocities, self.masses, self.particle_types, self.atom_types,
         self.molecule_indexes, self.electrical_charges, self.bonds,
         self.angles) = self.initialize_particle_properties()

        self.target_velocity_distributions = get_target_water_velocity_distributions(
            number_particles=self.number_water_molecules * 3,
            masses=self.masses,
            mass_dictionary=self.mass_dictionary,
            boltzman_constant=self.boltzman_constant,
            desired_temperature=self.desired_temperature)

    def _create_simulation_directory(self):
        """
        Creates the directory for the simulation output files.

        The directory is named based on the boundary condition and desired temperature.
        If the directory already exists, it is removed and recreated.
        """
        self.simulation_directory = self.simulation_configuration["simulation_directory"]
        os.makedirs(self.simulation_directory, exist_ok=True)
        self.simulation_destination = os.path.join(self.simulation_directory,
                                                   f"{self.boundary_condition}_{self.desired_temperature}K")

        if os.path.exists(self.simulation_destination):
            shutil.rmtree(self.simulation_destination)
        os.makedirs(self.simulation_destination, exist_ok=True)

    def initialize_particle_properties(self):
        """
        Initializes the properties of all particles (atoms) in the simulation.

        This method creates WaterMolecule objects and aggregates their properties,
        such as positions, velocities, masses, charges, bonds, and angles.

        Returns:
            tuple: A tuple containing numpy arrays for positions, velocities, masses,
                   particle types, atom types, molecule indexes, electrical charges,
                   and lists for bonds and angles.
        """
        positions = []
        velocities = []
        masses = []
        particle_types = []
        atom_types = []
        molecule_indexes = []
        electrical_charges = []
        bonds = []
        angles = []

        occupied_positions = []
        for i in range(self.number_water_molecules):
            water_molecule = WaterMolecule(
                molecule_id=i,
                occupied_positions=occupied_positions,
                simulation_box_size=self.simulation_box_size,
                position_initiation_configurations=self.simulation_configuration["position_initiation_configurations"],
                mass_configurations=self.simulation_configuration["mass_configurations"],
                particle_type_configurations=self.simulation_configuration["particle_type_configurations"],
                water_configurations=self.simulation_configuration["water_configurations"])

            occupied_positions.append(water_molecule.positions[1])
            positions.extend(water_molecule.positions)
            velocities.extend(water_molecule.velocities)
            masses.extend(water_molecule.masses)
            particle_types.extend(water_molecule.particle_types)
            atom_types.extend(water_molecule.atom_types)
            molecule_indexes.extend(water_molecule.molecule_indexes)
            electrical_charges.extend(water_molecule.atom_charges)
            bonds.extend(water_molecule.bonds)
            angles.extend(water_molecule.angles)

        positions = np.array(positions)
        velocities = np.array(velocities)
        masses = np.array(masses)
        particle_types = np.array(particle_types)
        atom_types = np.array(atom_types)
        molecule_indexes = np.array(molecule_indexes)
        electrical_charges = np.array(electrical_charges)

        return (positions, velocities, masses, particle_types, atom_types,
                molecule_indexes, electrical_charges, bonds, angles)

    def run_simulation(self):
        """
        Runs the molecular dynamics simulation.

        This method contains the main simulation loop, which iterates over the specified
        number of simulation steps. In each step, it calculates forces, updates velocities
        and positions, and writes the output to a file.
        """
        for iteration_index in tqdm(range(self.simulation_steps),
                                    desc=f"Running {self.boundary_condition}_{self.desired_temperature}K Simulation :"):
            self.velocities, _ = update_velocity_using_forces(
                positions=self.positions,
                velocities=self.velocities,
                bonds=self.bonds,
                sigma=self.lennard_jones_parameters["sigma"],
                epsilon=self.lennard_jones_parameters["epsilon"],
                angles=self.angles,
                molecule_indexes=self.molecule_indexes,
                electrical_charges=self.electrical_charges,
                coulombs_constant=self.coulombs_constant,
                time_step=self.time_step,
                masses=self.masses,
                atom_types=self.atom_types,
                dimensions=self.dimensions)

            if iteration_index % 25 == 0:
                self.velocities = correct_velocities_based_on_target_velocity_distributions(
                    velocities=self.velocities,
                    mass_dictionary=self.mass_dictionary,
                    boltzman_constant=self.boltzman_constant,
                    target_velocity_distributions=self.target_velocity_distributions)
            else:
                self.velocities = correct_velocities_based_on_temperature(
                    velocities=self.velocities,
                    masses=self.masses,
                    boltzman_constant=self.boltzman_constant,
                    desired_temperature=self.desired_temperature)

            self.positions, self.velocities = update_positions_and_velocities(
                positions=self.positions,
                velocities=self.velocities,
                simulation_box_size=self.simulation_box_size,
                boundary_conditions=self.boundary_condition,
                time_step=self.time_step)

            write_positions_to_file(
                positions=self.positions,
                simulation_box_size=self.simulation_box_size,
                simulation_directory=self.simulation_destination,
                iteration_index=iteration_index,
                particle_types=self.particle_types)
