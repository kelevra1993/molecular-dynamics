import os
from os.path import dirname
from utilities.utils import read_json
from simulator.simulator import Simulator


def main():
    configuration_folder = os.path.join(dirname(__file__), "configurations")
    simulation_configuration = read_json(os.path.join(configuration_folder, "simulation_configurations.json"))

    for desired_temperature in [200]:
        simulator = Simulator(number_water_molecules=simulation_configuration["number_water_molecules"],
                              simulation_configuration=simulation_configuration,
                              desired_temperature=desired_temperature,
                              boundary_condition="reflective")
        simulator.run_simulation()
    return None


if __name__ == "__main__":
    main()
