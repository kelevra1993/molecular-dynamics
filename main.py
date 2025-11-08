import os
from os.path import dirname
from utilities.utils import read_json, load_yaml_configuration, print_blue, print_green
from simulator.simulator import Simulator


def main():
    configuration_folder = os.path.join(dirname(__file__), "configurations")

    # Load Simulation Configurations
    simulation_configuration_path = os.path.join(configuration_folder, "simulation_configurations.yml")
    simulation_configuration = load_yaml_configuration(simulation_configuration_path)

    # For loop for desired temperatures of the simulation
    for desired_temperature in [200]:
        print_blue(f"Running Simulation For Temperature {desired_temperature} Kelvin", add_separators=True)
        simulator = Simulator(number_water_molecules=simulation_configuration["number_water_molecules"],
                              simulation_configuration=simulation_configuration,
                              desired_temperature=desired_temperature,
                              boundary_condition="reflective")
        simulator.run_simulation()
        print_green(f"Simulation Done For Temperature {desired_temperature} Kelvin", add_separators=True)
    return None


if __name__ == "__main__":
    main()
