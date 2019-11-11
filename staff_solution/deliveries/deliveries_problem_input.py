import os
from typing import *
from dataclasses import dataclass
from enum import Enum

from framework import *


__all__ = [
    'OptimizationTargetType', 'PackagesDelivery', 'DeliveriesTruck', 'DeliveriesTruckProblemInput'
]


class OptimizationTargetType(Enum):
    Distance = 'Distance'
    Time = 'Time'
    Money = 'Money'


@dataclass(frozen=True)
class PackagesDelivery:
    client_name: str
    pick_location: Junction
    drop_location: Junction
    nr_packages: int

    def serialize(self) -> str:
        return f'{self.client_name},{self.pick_location.index},{self.drop_location.index},{self.nr_packages}'

    @staticmethod
    def deserialize(serialized: str, roads: Roads) -> 'PackagesDelivery':
        parts = serialized.split(',')
        return PackagesDelivery(
            client_name=parts[0],
            pick_location=roads[int(parts[1])],
            drop_location=roads[int(parts[2])],
            nr_packages=int(parts[3]))

    def __repr__(self):
        return str(self.client_name)


@dataclass(frozen=True)
class DeliveriesTruck:
    max_nr_loaded_packages: int
    initial_location: Junction
    optimal_vehicle_speed: float = kmh_to_ms(87)
    gas_cost_per_meter_in_optimal_speed: float = 0.7
    gas_cost_per_meter_gradient: float = 0.2
    gas_cost_addition_per_meter_per_loaded_package: float = 0

    def serialize(self) -> str:
        return f'{self.max_nr_loaded_packages},{self.initial_location.index}'

    @staticmethod
    def deserialize(serialized: str, roads: Roads) -> 'DeliveriesTruck':
        parts = serialized.split(',')
        max_nr_loaded_packages, initial_location_index = int(parts[0]), int(parts[1])
        return DeliveriesTruck(
            max_nr_loaded_packages=max_nr_loaded_packages,
            initial_location=roads[initial_location_index])

    def calc_optimal_driving_parameters(self, target_type: OptimizationTargetType, max_driving_speed: float) -> Tuple[float, float]:
        if target_type == OptimizationTargetType.Time or target_type == OptimizationTargetType.Distance:
            optimal_driving_speed = max_driving_speed
        else:
            assert target_type == OptimizationTargetType.Money
            optimal_driving_speed = self.optimal_vehicle_speed if self.optimal_vehicle_speed < max_driving_speed else max_driving_speed
        gas_cost_per_meter = self.gas_cost_per_meter_in_optimal_speed + \
                             self.gas_cost_per_meter_gradient * (abs(optimal_driving_speed - self.optimal_vehicle_speed) / self.optimal_vehicle_speed)
        return optimal_driving_speed, gas_cost_per_meter


@dataclass(frozen=True)
class DeliveriesTruckProblemInput:
    input_name: str
    deliveries: Tuple[PackagesDelivery, ...]
    delivery_truck: DeliveriesTruck
    toll_road_cost_per_meter: float = 1.2  # TODO: set a meaningful value here; TODO: add to input file (load&store) & to generate script.

    @staticmethod
    def load_from_file(input_file_name: str, roads: Roads) -> 'DeliveriesTruckProblemInput':
        """
        Loads and parses a deliveries-problem-input from a file. Usage example:
        >>> problem_input = DeliveriesTruckProblemInput.load_from_file('big_delivery.in', roads)
        """

        with open(Consts.get_data_file_path(input_file_name), 'r') as input_file:
            input_type = input_file.readline().strip()
            if input_type != 'DeliveriesTruckProblemInput':
                raise ValueError(f'Input file `{input_file_name}` is not a deliveries input.')
            try:
                input_name = input_file.readline().strip()
                deliveries = tuple(
                    PackagesDelivery.deserialize(serialized_delivery, roads)
                    for serialized_delivery in input_file.readline().rstrip('\n').split(';'))
                delivery_truck = DeliveriesTruck.deserialize(input_file.readline().rstrip('\n'), roads)
            except:
                raise ValueError(f'Invalid input file `{input_file_name}`.')
        return DeliveriesTruckProblemInput(input_name=input_name, deliveries=deliveries, delivery_truck=delivery_truck)

    def store_to_file(self, input_file_name: str):
        with open(Consts.get_data_file_path(input_file_name), 'w') as input_file:
            lines = [
                'DeliveriesTruckProblemInput',
                str(self.input_name.strip()),
                ';'.join(delivery.serialize() for delivery in self.deliveries),
                self.delivery_truck.serialize()
            ]
            for line in lines:
                input_file.write(line + '\n')

    @staticmethod
    def load_all_inputs(roads: Roads) -> Dict[str, 'DeliveriesTruckProblemInput']:
        """
        Loads all the inputs in the inputs directory.
        :return: list of inputs.
        """
        inputs = {}
        input_file_names = [f for f in os.listdir(Consts.DATA_PATH)
                            if os.path.isfile(os.path.join(Consts.DATA_PATH, f)) and f.split('.')[-1] == 'in']
        for input_file_name in input_file_names:
            try:
                problem_input = DeliveriesTruckProblemInput.load_from_file(input_file_name, roads)
                inputs[problem_input.input_name] = problem_input
            except:
                pass
        return inputs
