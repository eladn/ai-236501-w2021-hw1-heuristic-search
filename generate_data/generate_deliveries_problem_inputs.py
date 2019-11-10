import os
import sys

if os.path.basename(os.getcwd().rstrip('/')) == 'generate_data':
    new_path = os.path.join(os.getcwd(), '../staff_solution/')
    if os.getcwd() in sys.path:
        sys.path.remove(os.getcwd())
    sys.path.append(new_path)
    os.chdir(new_path)
print(os.getcwd())

from framework import *
from deliveries import DeliveriesProblemInput

import numpy as np
from typing import Tuple, FrozenSet, Union


def generate_deliveries_problem_input_junctions(
        roads: Roads,
        nr_drop_points: int,
        nr_gas_stations: int,
        choosing_junctions_seed: int = 0,
        limit_to_radius: Union[float, None] = None) -> Tuple[Junction, FrozenSet[Junction], FrozenSet[Junction]]:
    rnd = np.random.RandomState(choosing_junctions_seed)

    all_junction_indeces = list(roads.keys())
    if limit_to_radius:
        center_junction_idx = rnd.choice(all_junction_indeces, 1)[0]
        center_junction = roads[center_junction_idx]
        all_junction_indeces = [
            junction_idx
            for junction_idx, junction in roads.items()
            if center_junction.calc_air_distance_from(junction) <= limit_to_radius
        ]

    nr_choise = 1 + nr_drop_points + nr_gas_stations
    all_junction_indeces = np.array(all_junction_indeces)
    chosen_junction_idxs = rnd.choice(all_junction_indeces, nr_choise, replace=False)
    assert all(idx in roads for idx in chosen_junction_idxs)
    start_point = roads[chosen_junction_idxs[0]]
    drop_points = frozenset({roads[idx] for idx in chosen_junction_idxs[1:nr_drop_points+1]})
    gas_stations = frozenset({roads[idx] for idx in chosen_junction_idxs[1+nr_drop_points:]})
    return start_point, drop_points, gas_stations


def generate_supplied_deliveries_problem_inputs_files(roads: Roads):
    inputs = [
        DeliveriesProblemInput('small_delivery',
                               *generate_deliveries_problem_input_junctions(roads, 5, 8, 0x5739572, 6000),
                               7000, 7000),
        DeliveriesProblemInput('big_delivery',
                               *generate_deliveries_problem_input_junctions(roads, 8, 30, 2342489),
                               20000, 20000),
    ]
    for problem_input in inputs:
        problem_input.store_to_file(problem_input.input_name + '.in')


def generate_test_deliveries_problem_inputs_files(roads: Roads):
    # TODO: change seeds, junctions, tank-capacities!
    inputs = [
        DeliveriesProblemInput('test1_small_delivery',
                               *generate_deliveries_problem_input_junctions(roads, 5, 9, 0x82e67c83 % 2**32, 5000),
                               7000, 7000),
        DeliveriesProblemInput('test1_big_delivery',
                               *generate_deliveries_problem_input_junctions(roads, 8, 30, 0x2bd883a83e % 2**32),
                               20000, 20000),
    ]
    for problem_input in inputs:
        problem_input.store_to_file(problem_input.input_name + '.in')


def data_generation_test(roads: Roads):
    """Just for sanity checks of generations"""

    from staff_aux.staff_solution.deliveries import DeliveriesProblemInput, RelaxedDeliveriesProblem, \
        StrictDeliveriesProblem, MSTAirDistHeuristic, AirDistHeuristic

    inputs = DeliveriesProblemInput.load_all_inputs(roads)

    for problem_input in inputs:
        from itertools import combinations
        all_points = problem_input.drop_points | problem_input.gas_stations
        max_dist = max(junc1.calc_air_distance_from(junc2) for junc1, junc2 in combinations(all_points, r=2))
        print(problem_input.input_name, 'max dist between points:', max_dist)

    for problem_input in inputs:
        relaxed_problem = RelaxedDeliveriesProblem(problem_input)
        strict_problem = StrictDeliveriesProblem(problem_input, roads, AStar(AirDistHeuristic))
        astar = AStar(MSTAirDistHeuristic)
        res = astar.solve_problem(relaxed_problem)
        print(res)
        if len(problem_input.drop_points) <= 5:
            res = astar.solve_problem(strict_problem)
            print(res)


if __name__ == '__main__':
    roads = load_map_from_csv(Consts.get_data_file_path('tlv.csv'))
    generate_supplied_deliveries_problem_inputs_files(roads)
    generate_test_deliveries_problem_inputs_files(roads)
    data_generation_test(roads)
