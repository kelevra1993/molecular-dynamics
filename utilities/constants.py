"""
# TODO To be documented
"""

# Simulation Destination
simulation_directory = "/Users/Robert/PycharmProjects/molecular-dynamics/simulations"

# Simulation Global Parameters
number_particles = 20
dimensions = 3
time_step = 0.005  # timesteps in pico-seconds so 1e-12
simulation_steps = 5000
simulation_box_size = 50  # System in Angstom
desired_temperature = 1000
# Lennard-Jones Interaction Parameters
# For hydrogen
# Sigma In angstrom
lennard_jones_paramaters = {"epsilon": 3.682, "sigma": 2.928}

# Partical Masses in Dalton
mass_dictionary = {"hydrogen": 1, "nitrogen": 14, "carbon": 12, "oxygen": 16}

# How to keep energy constant in the system to avoid energy increasing ?
# We will keep temperature constant
# In order to keep everything simple we will be setting boltzman constant Kb to Angstroms, Daltons, Pico-seconds

# Original Kb
boltzman_constant = 1.380649E-23  # In SI, this is equivalent to [m2] [kg] [s-2] [K-1]
boltzman_constant = boltzman_constant * 6.02228E26  # Turning kg -> Da since we know 1Da = 1,6605E-27 Kg so [m2] [Da] [s-2] [K-1]
boltzman_constant = boltzman_constant * 1E20  # Turning m -> Angstrom2(A)  sine we know 1Da = 1,6605E-27 Kg so [A^2] [Da] [s-2] [K-1]
boltzman_constant = boltzman_constant / 1E24  # Turning seconds -> picosends(A^2)  sine we know 1Da = 1,6605E-27 Kg so [A^2] [Da] [s-2] [K-1]  # Kb = 0.8314654859720001
