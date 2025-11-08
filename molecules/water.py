import numpy as np


class WaterMolecule:
    """
    Represents a single water molecule in the simulation, including its atoms' positions, velocities, and properties.
    """

    def __init__(self, molecule_id, occupied_positions, simulation_box_size, max_occupancy, initial_hydrogen_offset,
                 distance_maximization_step, minimum_desired_distance, mass_configurations,
                 particle_type_configurations, water_configuration):
        """
        Initializes a WaterMolecule object.

        Args:
            molecule_id (int): A unique identifier for the molecule.
            occupied_positions (list): A list of positions already occupied by other molecules.
            simulation_box_size (float): The size of the simulation box.
            max_occupancy (float): The maximum occupancy of the simulation box.
            initial_hydrogen_offset (float): The initial offset for hydrogen atoms from the oxygen atom.
            distance_maximization_step (int): The number of steps to take to maximize the distance between molecules.
            minimum_desired_distance (float): The minimum desired distance between molecules.
            mass_configurations (dict): A dictionary of mass configurations for different atom types.
            particle_type_configurations (dict): A dictionary of particle type configurations.
            water_configuration (dict): A dictionary of configuration parameters for water molecules.
        """
        self.simulation_box_size = simulation_box_size
        self.occupied_positions = occupied_positions
        self.molecule_id = molecule_id
        self.max_occupancy = max_occupancy
        self.initial_hydrogen_offset = initial_hydrogen_offset
        self.distance_maximization_step = distance_maximization_step
        self.minimum_desired_distance = minimum_desired_distance
        self.mass_configurations = mass_configurations
        self.particle_type_configurations = particle_type_configurations
        self.water_configuration = water_configuration
        self.atom_charge_dictionary = {"water": {"oxygen": -0.82, "hydrogen": 0.41}}

        self.positions = self.get_water_positions()
        self.velocities = self.get_water_velocities()
        self.masses = self.get_particle_mass()
        self.particle_types = self.get_particle_type()
        self.atom_types = self.get_atom_types()
        self.atom_charges = self.get_atom_charges()
        self.bonds = self.get_water_bonds()
        self.angles = self.get_water_angles()

    def get_water_positions(self):
        """
        Generates the positions of the three atoms of a water molecule (H, O, H).
        The oxygen atom is placed randomly, and the hydrogen atoms are placed around it.
        It ensures that the new molecule does not overlap with existing ones.

        Returns:
            list: A list of three numpy arrays representing the positions of the hydrogen, oxygen, and hydrogen atoms.
        """
        # Scale the simulation box to the desired occupancy
        target_box_size = int(self.max_occupancy * self.simulation_box_size)
        center_factor = (1 - self.max_occupancy) / 2
        translator_vector = center_factor * self.simulation_box_size * np.ones(3)

        box_dimensions = np.array([target_box_size] * 3)

        # Generate a random position for the Oxygen atom
        oxygen_position = np.random.rand(3) * box_dimensions

        # If there are other molecules, try to place the new one as far as possible from them
        if not self.occupied_positions:
            self.occupied_positions.append(oxygen_position)
        else:
            least_worst_oxygen_position = {"position": [], "distance": 0}
            # Try to find a better position in a limited number of steps
            for distance_step in range(self.distance_maximization_step):
                vectors_to_atom_i = oxygen_position - self.occupied_positions
                distance_norms = np.linalg.norm(vectors_to_atom_i, axis=1)

                if np.min(distance_norms) > least_worst_oxygen_position["distance"]:
                    least_worst_oxygen_position = {"position": oxygen_position, "distance": np.min(distance_norms)}

                number_of_close_oxygens = sum(x < self.minimum_desired_distance for x in distance_norms)

                if number_of_close_oxygens:
                    oxygen_position = np.random.rand(3) * box_dimensions
                else:
                    break
                if distance_step == self.distance_maximization_step - 1:
                    oxygen_position = least_worst_oxygen_position["position"]

        self.occupied_positions.append(oxygen_position)

        # Calculate the positions of the two Hydrogen atoms around the Oxygen
        random_angle = np.random.rand() * 2.0 * np.pi
        cos_theta = np.cos(random_angle)
        sin_theta = np.sin(random_angle)

        h1_offset = np.array([self.initial_hydrogen_offset * cos_theta, self.initial_hydrogen_offset * sin_theta, 0.0])
        h2_offset = np.array([-self.initial_hydrogen_offset * sin_theta, self.initial_hydrogen_offset * cos_theta, 0.0])

        hydrogen1_position = oxygen_position + h1_offset
        hydrogen2_position = oxygen_position + h2_offset

        # Center the molecule in the simulation box and return the positions
        positions = [hydrogen1_position + translator_vector, oxygen_position + translator_vector,
                     hydrogen2_position + translator_vector]

        return positions

    def get_water_velocities(self):
        """
        Generates the velocities of the three atoms of a water molecule.
        All atoms in the molecule are given the same initial velocity.

        Returns:
            list: A list of three numpy arrays representing the velocities of the atoms.
        """
        oxygen_velocity = (np.random.rand(3) - 0.5) * self.simulation_box_size
        velocities = [oxygen_velocity, oxygen_velocity, oxygen_velocity]
        return velocities

    def get_particle_mass(self):
        """
        Returns the masses of the atoms in the water molecule.

        Returns:
            list: A list of masses for [hydrogen, oxygen, hydrogen].
        """
        mass_hydrogen = self.mass_configurations["hydrogen"]
        mass_oxygen = self.mass_configurations["oxygen"]
        return [mass_hydrogen, mass_oxygen, mass_hydrogen]

    def get_particle_type(self):
        """
        Returns the particle types of the atoms in the water molecule.

        Returns:
            list: A list of particle types for [hydrogen, oxygen, hydrogen].
        """
        type_hydrogen = self.particle_type_configurations["hydrogen"]
        type_oxygen = self.particle_type_configurations["oxygen"]
        return [type_hydrogen, type_oxygen, type_hydrogen]

    @staticmethod
    def get_atom_types(self):
        """
        Returns the atom types of the atoms in the water molecule.

        Returns:
            list: A list of atom types for [hydrogen, oxygen, hydrogen].
        """
        return ["hydrogen", "oxygen", "hydrogen"]

    def get_atom_charges(self):
        """
        Returns the partial charges of the atoms in the water molecule.

        Returns:
            list: A list of partial charges for [hydrogen, oxygen, hydrogen].
        """
        charge_hydrogen = self.atom_charge_dictionary["water"]["hydrogen"]
        charge_oxygen = self.atom_charge_dictionary["water"]["oxygen"]
        return [charge_hydrogen, charge_oxygen, charge_hydrogen]

    def get_water_bonds(self):
        """
        Defines the bonds between the atoms of the water molecule.

        Returns:
            list: A list of bonds, where each bond is a list of [atom1_index, atom2_index, bond_length, spring_constant].
        """
        bond_length = self.water_configuration["bond_length"]
        bond_spring_constant = self.water_configuration["water_bond_spring_constant"]
        i = self.molecule_id
        bonds = [[3 * i, 3 * i + 1, bond_length, bond_spring_constant],
                 [3 * i + 1, 3 * i + 2, bond_length, bond_spring_constant]]
        return bonds

    def get_water_angles(self):
        """
        Defines the angle of the water molecule.

        Returns:
            list: A list containing the angle information [atom1_index, atom2_index, atom3_index, angle, spring_constant].
        """
        water_angle = self.water_configuration["water_angle"]
        angle_spring_constant = self.water_configuration["water_angle_spring_constant"]
        i = self.molecule_id
        angles = [[3 * i, 3 * i + 1, 3 * i + 2, water_angle, angle_spring_constant]]
        return angles
