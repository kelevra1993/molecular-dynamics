"""
# TODO To be documented
"""
import numpy as np

# Simulation Destination
simulation_directory = "/Users/Robert/PycharmProjects/molecular-dynamics/simulations"

# Simulation Global Parameters
number_particles = 60
# todo work on removing this to always work with 3 dimensions
dimensions = 3
# time_step = 0.0001  # timesteps in pico-seconds so 1e-12
time_step = 0.001  # timesteps in pico-seconds so 1e-12
simulation_steps = 10000
simulation_box_size = 50  # System in Angstom
desired_temperatures = [1000]
# Lennard-Jones Interaction Parameters
# For hydrogen
# Sigma In angstrom
lennard_jones_parameters = {"epsilon": {"oxygen_oxygen": 62.87, "oxygen_hydrogen": 14.2723, "hydrogen_hydrogen": 3.24},
                            "sigma": {"oxygen_oxygen": 3.1169, "oxygen_hydrogen": 2.04845, "hydrogen_hydrogen": 0.98}}
# Partical Masses in Dalton
mass_dictionary = {"hydrogen": 1, "nitrogen": 14, "carbon": 12, "oxygen": 16}

# Atome Charge Dictionary
atome_charge_dictionary = {"water": {"oxygen": -0.82, "hydrogen": 0.41}}

# Water bond spring constant
water_bond_spring_constant = 148000.0 / 2

# Desired water bond length
water_bond_length = 0.9611

# Water angle spring constant
water_angle_spring_constant = 35300.0 / 2

# Desired water angle
water_angle = 104.45 * np.pi / 180.0  # Angles in Radian

# How to keep energy constant in the system to avoid energy increasing ?
# We will keep temperature constant
# In order to keep everything simple we will be setting boltzman constant Kb to Angstroms, Daltons, Pico-seconds

# Original Kb
# In SI, this is equivalent to [m2] [kg] [s-2] [K-1]
boltzman_constant = 1.380649E-23
# Turning kg -> Da since we know 1Da = 1,6605E-27 Kg so [m2] [Da] [s-2] [K-1]
boltzman_constant = boltzman_constant * 6.02228E26
# Turning m -> Angstrom2(A)  sine we know 1Da = 1,6605E-27 Kg so [A^2] [Da] [s-2] [K-1]
boltzman_constant = boltzman_constant * 1E20
# Turning seconds -> picosends(A^2)  sine we know 1Da = 1,6605E-27 Kg so [A^2] [Da] [s-2] [K-1]  # Kb = 0.8314654859720001
boltzman_constant = boltzman_constant / 1E24

# Coulombs constant
avagadros_constant_g_wise = 6.02228E26  # Avogardos constant x 1000 (g->kg)
electron_charge = 1.60217662E-19  # electron charge in coulombs
coulombs_constant = 8.9875517923E9 * avagadros_constant_g_wise * 1E30 * electron_charge * electron_charge / 1E24  # electrostatic constant in Daltons, electron charges, picosecond, angstrom units
