from framework import *
from deliveries import *

import numpy as np
from typing import *
from warnings import warn


PERSON_FULL_NAMES = """
Doris Sollers
Maryjane Dickert
Mindy Drennen
Kyle Turck
Lannie Hunkins
Krysta Valentine
Ethan Mulholland
Cleora Alaniz
Leia Urquhart
Natalia Templin
Jasmine Denson
Quinton Salais
Raven Woolum
Gayle Schild
Hobert Venable
Anisha Decker
Johnny Fierro
Tess Bish
Christopher Dickinson
Kurt Dockstader
Hana Hockman
Stefani Kreutzer
Nettie Eggebrecht
Leif Haug
Meda Geisler
Cyril Boyland
Gussie Foran
Margorie Bouldin
Lizzette Kohn
Hedy Moser
Kori Satcher
Kirstie Demyan
Lyla Motto
Sidney Dunmire
Arron Mungin
Willie Luellen
Max Neri
Tegan Storms
Stacia Landsman
Marlen Ehle
Julia Hadlock
Pierre Lowman
Veronique Katz
Debi Basta
Edie Echevarria
Rhett Klopfenstein
Felicitas Cardone
Elicia Klemme
Joaquina Olsson
Jeane Ravencraft
""".strip().split('\n')
PERSON_FULL_NAMES = [name.strip() for name in PERSON_FULL_NAMES if name.strip()]


def generate_deliveries_problem_input_junctions(
        roads: StreetsMap,
        nr_junctions: int,
        random_state: np.random.RandomState,
        limit_to_radius: Optional[float] = None) -> List[Junction]:
    all_junction_indices = list(roads.keys())
    if limit_to_radius:
        center_junction_idx = random_state.choice(all_junction_indices, 1)[0]
        center_junction = roads[center_junction_idx]
        all_junction_indices = [
            junction_idx
            for junction_idx, junction in roads.items()
            if center_junction.calc_air_distance_from(junction) <= limit_to_radius
        ]

    assert nr_junctions < 0.2 * len(all_junction_indices)
    all_junction_indices = np.array(all_junction_indices)
    chosen_junction_idxs = random_state.choice(all_junction_indices, nr_junctions, replace=False)
    assert all(idx in roads for idx in chosen_junction_idxs)
    assert len(set(chosen_junction_idxs)) == nr_junctions
    return [roads[idx] for idx in chosen_junction_idxs]


# def generate_multiple_agents_pizzas_network_deliveries_problem_input(
#         input_name: str,
#         roads: StreetsMap,
#         nr_orders: int,
#         nr_pizza_places: int,
#         delivery_vehicles_without_start_location: Tuple[DeliveriesVehicle, ...],
#         choosing_junctions_seed: int = 0,
#         limit_to_radius: Optional[float] = None) -> DeliveriesProblemInput:
#     random_state = np.random.RandomState(choosing_junctions_seed)
#     nr_delivery_vehicles = len(delivery_vehicles_without_start_location)
#     all_sampled_junctions = generate_deliveries_problem_input_junctions(
#         roads=roads,
#         nr_junctions=nr_orders+nr_pizza_places+nr_delivery_vehicles,
#         random_state=random_state, limit_to_radius=limit_to_radius
#     )
#
#     orders_junctions = all_sampled_junctions[0:nr_orders]
#     assert len(orders_junctions) == nr_orders
#     pizza_places_junctions = all_sampled_junctions[nr_orders:nr_orders+nr_pizza_places]
#     assert len(pizza_places_junctions) == nr_pizza_places
#     delivery_vehicles_start_location_junctions = all_sampled_junctions[nr_orders+nr_pizza_places:]
#     assert len(delivery_vehicles_start_location_junctions) == nr_delivery_vehicles
#
#     orders = tuple(
#         PizzaOrder(order_id=order_id, location=order_junction,
#                    nr_pizzas=random_state.choice([1, 2, 3], p=[0.55, 0.3, 0.15]))
#         for order_id, order_junction in enumerate(orders_junctions)
#     )
#
#     pizza_stores = tuple(
#         PizzaStore(store_id=store_id, location=pizza_place_junction,
#                    max_nr_visits=random_state.choice([2, 3, 4], p=[0.5, 0.4, 0.1]))  # random_state.choice([2, 3, 4], p=[0.5, 0.4, 0.1]))
#         for store_id, pizza_place_junction in enumerate(pizza_places_junctions)
#     )
#
#     deliveries_vehicles = tuple(
#         DeliveriesVehicle(**{**delivery_vehicle.__dict__, 'start_location': start_location})
#         for delivery_vehicle, start_location
#         in zip(delivery_vehicles_without_start_location, delivery_vehicles_start_location_junctions)
#     )
#
#     return DeliveriesProblemInput(
#         input_name=input_name,
#         orders=orders,
#         pizza_stores=pizza_stores,
#         deliveries_vehicles=deliveries_vehicles
#     )


# def generate_supplied_multiple_agents_pizzas_network_deliveries_problem_inputs_files(roads: StreetsMap):
#     inputs = [
#         generate_multiple_agents_pizzas_network_deliveries_problem_input(
#             input_name='deliveries_problem_input',
#             roads=roads,
#             nr_orders=10,
#             nr_pizza_places=2,
#             delivery_vehicles_without_start_location=(
#                 # DeliveriesVehicle(
#                 #     name='Drony',
#                 #     vehicle_type=DeliveriesVehicleType.Drone,
#                 #     fuel_tank_capacity=10000,
#                 #     fuel_tank_initial_level=7000,
#                 #     pizzas_capacity=1,
#                 #     start_location=None
#                 # ),
#                 DeliveriesVehicle(
#                     name='Giyora',
#                     vehicle_type=DeliveriesVehicleType.Car,
#                     fuel_tank_capacity=200000,
#                     fuel_tank_initial_level=11000,
#                     pizzas_capacity=10,
#                     start_location=None
#                 ),
#                 DeliveriesVehicle(
#                     name='Valentin',
#                     vehicle_type=DeliveriesVehicleType.Motorcycle,
#                     fuel_tank_capacity=150000,
#                     fuel_tank_initial_level=11000,
#                     pizzas_capacity=2,
#                     start_location=None
#                 ),
#             ),
#             choosing_junctions_seed=0x5739574,
#             limit_to_radius=6000)
#     ]
#     for problem_input in inputs:
#         problem_input.store_to_file(problem_input.input_name + '.in')


def generate_truck_deliveries_problem_input(
        roads: StreetsMap,
        input_name: str = 'packages_truck_deliveries_problem_input',
        choosing_junctions_seed: int = 0x5739574,
        limit_to_radius: int = 6000,
        nr_deliveries: int = 5,
        max_nr_loaded_packages_in_truck: int = 7,
        nr_packages_options: Tuple[int, ...] = (2, 3, 4, 5),
        nr_packages_probabilities: Tuple[float, ...] = (0.2, 0.3, 0.2, 0.3)) -> DeliveriesTruckProblemInput:

    random_state = np.random.RandomState(choosing_junctions_seed)
    all_sampled_junctions = generate_deliveries_problem_input_junctions(
        roads=roads,
        nr_junctions=2 * nr_deliveries + 1,
        random_state=random_state, limit_to_radius=limit_to_radius
    )

    deliveries_pick_junctions = all_sampled_junctions[0:nr_deliveries]
    assert len(deliveries_pick_junctions) == nr_deliveries
    deliveries_drop_junctions = all_sampled_junctions[nr_deliveries:2*nr_deliveries]
    assert len(deliveries_drop_junctions) == nr_deliveries
    initial_truck_location = all_sampled_junctions[2*nr_deliveries]

    names = random_state.permutation(PERSON_FULL_NAMES)
    if nr_deliveries > len(names):
        warn(f'More deliveries to generate ({nr_deliveries}) than names ({len(names)}).')

    return DeliveriesTruckProblemInput(
        input_name=input_name,
        deliveries=tuple(
            Delivery(
                delivery_id=delivery_idx + 1,
                client_name=names[delivery_idx % len(names)],
                pick_location=pick_junction,
                drop_location=drop_junction,
                nr_packages=random_state.choice(nr_packages_options, p=nr_packages_probabilities)
            )
            for delivery_idx, (pick_junction, drop_junction)
            in enumerate(zip(deliveries_pick_junctions, deliveries_drop_junctions))
        ),
        delivery_truck=DeliveriesTruck(
            max_nr_loaded_packages=max_nr_loaded_packages_in_truck,
            initial_location=initial_truck_location),
        toll_road_cost_per_meter=0.001  # TODO: set a meaningful value here
    )


def generate_supplied_truck_deliveries_problem_inputs_files(roads: StreetsMap):
    inputs = [
        generate_truck_deliveries_problem_input(
            roads,
            input_name='small_delivery',
            choosing_junctions_seed=0x5739574,
            limit_to_radius=6000,
            nr_deliveries=5,
            max_nr_loaded_packages_in_truck=9,
            nr_packages_options=(2, 3, 4, 5),
            nr_packages_probabilities=(0.2, 0.3, 0.2, 0.3)),
        generate_truck_deliveries_problem_input(
            roads,
            input_name='moderate_delivery',
            choosing_junctions_seed=0x5739574,
            limit_to_radius=6000,
            nr_deliveries=8,
            max_nr_loaded_packages_in_truck=9,
            nr_packages_options=(2, 3, 4, 5),
            nr_packages_probabilities=(0.2, 0.3, 0.2, 0.3)),
        generate_truck_deliveries_problem_input(
            roads,
            input_name='big_delivery',
            choosing_junctions_seed=0x5739574,
            limit_to_radius=6000,
            nr_deliveries=15,
            max_nr_loaded_packages_in_truck=9,
            nr_packages_options=(2, 3, 4, 5),
            nr_packages_probabilities=(0.2, 0.3, 0.2, 0.3)),
        generate_truck_deliveries_problem_input(
            roads,
            input_name='test_deliveries_small',
            choosing_junctions_seed=0x2482424,
            limit_to_radius=6000,
            nr_deliveries=5,
            max_nr_loaded_packages_in_truck=9,
            nr_packages_options=(2, 3, 4, 5),
            nr_packages_probabilities=(0.2, 0.3, 0.2, 0.3)),
        generate_truck_deliveries_problem_input(
            roads,
            input_name='test_deliveries_medium',
            choosing_junctions_seed=0x2482424,
            limit_to_radius=6000,
            nr_deliveries=8,
            max_nr_loaded_packages_in_truck=9,
            nr_packages_options=(2, 3, 4, 5),
            nr_packages_probabilities=(0.2, 0.3, 0.2, 0.3)),
    ]
    for problem_input in inputs:
        problem_input.store_to_file(problem_input.input_name + '.in')


def generate_test_deliveries_problem_inputs_files(roads: StreetsMap):
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


def data_generation_test(roads: StreetsMap):
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


def update_streets_map():
    print('Generating new streets-map with toll roads and max speed per road ..')
    streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path('tlv_original.csv'))
    print(f'#links: {sum(1 for _ in streets_map.iterlinks())} (before remove dangling links)')
    streets_map.remove_dangling_links()
    print(f'#links: {sum(1 for _ in streets_map.iterlinks())} (after remove dangling links, before remove zero dist links)')
    streets_map.remove_zero_distance_links()
    print(f'#links: {sum(1 for _ in streets_map.iterlinks())} (after remove dangling links, after remove zero dist links)')
    streets_map.update_link_distances_to_air_distance()
    streets_map.set_links_max_speed_and_is_toll()
    streets_map.set_incoming_links()
    streets_map.write_to_csv(Consts.get_data_file_path('tlv_streets_map.csv'))
    print('Done generating the new streets-map.')


if __name__ == '__main__':
    # update_streets_map()
    # exit()
    streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path('tlv_streets_map.csv'))
    generate_supplied_truck_deliveries_problem_inputs_files(streets_map)
    # generate_supplied_multiple_agents_pizzas_network_deliveries_problem_inputs_files(streets_map)
    # generate_test_deliveries_problem_inputs_files(streets_map)
    # data_generation_test(streets_map)
