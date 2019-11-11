import os
import itertools
from typing import *
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree as mst

from framework import *
from .map_problem import MapProblem
from .cached_map_distance_finder import CachedMapDistanceFinder


@dataclass(frozen=True)
class PizzaStore:
    store_id: int
    location: Junction
    max_nr_visits: int
    # TODO: add `has_gas_station: boolean`

    @staticmethod
    def deserialize(serialized_pizza_store: str, roads: Roads) -> 'PizzaStore':
        parts = serialized_pizza_store.split(',')
        return PizzaStore(store_id=int(parts[0]), location=roads[int(parts[1])], max_nr_visits=int(parts[2]))

    def serialize(self) -> str:
        return f'{self.store_id},{self.location.index},{self.max_nr_visits}'

    def __repr__(self):
        return f'PizzaStore({self.store_id})'


@dataclass(frozen=True)
class PizzaOrder:
    order_id: int
    location: Junction
    nr_pizzas: int

    @staticmethod
    def deserialize(serialized_order: str, roads: Roads) -> 'PizzaOrder':
        parts = serialized_order.split(',')
        return PizzaOrder(order_id=int(parts[0]), location=roads[int(parts[1])], nr_pizzas=int(parts[2]))

    def serialize(self) -> str:
        return f'{self.order_id},{self.location.index},{self.nr_pizzas}'

    def __repr__(self):
        return f'PizzaOrder({self.order_id})'


class DeliveriesVehicleType(Enum):
    Motorcycle = 'Motorcycle'
    Car = 'Car'
    Drone = 'Drone'

    @staticmethod
    def deserialize(deliveries_vehicle_type_str: str) -> 'DeliveriesVehicleType':
        return DeliveriesVehicleType.__members__[deliveries_vehicle_type_str]

    def serialize(self):
        return self.value


@dataclass(frozen=True)
class DeliveriesVehicle:
    name: str
    vehicle_type: DeliveriesVehicleType
    fuel_tank_capacity: float
    fuel_tank_initial_level: float
    pizzas_capacity: int
    start_location: Junction

    @staticmethod
    def deserialize(serialized_deliveries_vehicle: str, roads: Roads) -> 'DeliveriesVehicle':
        parts = serialized_deliveries_vehicle.split(',')
        return DeliveriesVehicle(
            name=str(parts[0]),
            vehicle_type=DeliveriesVehicleType.deserialize(str(parts[1])),
            fuel_tank_capacity=float(parts[2]),
            fuel_tank_initial_level=float(parts[3]),
            pizzas_capacity=int(parts[4]),
            start_location=roads[int(parts[5])]
        )

    def serialize(self) -> str:
        return f'{self.name},{self.vehicle_type.serialize()},{self.fuel_tank_capacity},' \
               f'{self.fuel_tank_initial_level},{self.pizzas_capacity},{self.start_location.index}'


@dataclass(frozen=True)
class DeliveriesVehicleState:
    current_location: Union[Junction, PizzaOrder, PizzaStore]
    # fuel_level: float
    nr_loaded_pizzas: int
    dropped_so_far: FrozenSet[PizzaOrder]
    # distance_travel_meter: float

    def get_current_junction(self):
        if isinstance(self.current_location, Junction):
            return self.current_location
        if isinstance(self.current_location, PizzaOrder) or isinstance(self.current_location, PizzaStore):
            return self.current_location.location


@dataclass(frozen=True)
class PizzaStoreState:
    nr_visits: int


@dataclass(frozen=True)
class DeliveriesState(GraphProblemState):
    """
    TODO: write doc here
    """
    vehicles: Tuple[DeliveriesVehicleState]
    pizza_stores: Tuple[PizzaStoreState]

    @property
    def dropped_so_far(self) -> FrozenSet[PizzaOrder]:
        return frozenset(set().union(*(vehicle.dropped_so_far for vehicle in self.vehicles)))

    # @property
    # def max_traveled_distance_so_far(self):
    #     return max(*(vehicle_state.distance_travel_meter for vehicle_state in self.vehicles))

    def __eq__(self, other):
        assert isinstance(other, DeliveriesState)
        return self.vehicles == other.vehicles

    def __hash__(self):
        return hash(self.vehicles)

    def __str__(self):
        return str(self.vehicles)


@dataclass(frozen=True)
class DeliveriesProblemInput:
    """
    This class is used to store and represent the input parameters
    to a deliveries-problem.
    It has a static method that may be used to load an input from a file. Usage example:
    >>> problem_input = DeliveriesProblemInput.load_from_file('big_delivery.in', roads)
    """

    input_name: str
    orders: Tuple[PizzaOrder]
    pizza_stores: Tuple[PizzaStore]
    deliveries_vehicles: Tuple[DeliveriesVehicle]

    @staticmethod
    def load_from_file(input_file_name: str, roads: Roads) -> 'DeliveriesProblemInput':
        """
        Loads and parses a deliveries-problem-input from a file. Usage example:
        >>> problem_input = DeliveriesProblemInput.load_from_file('big_delivery.in', roads)
        """

        with open(Consts.get_data_file_path(input_file_name), 'r') as input_file:
            input_type = input_file.readline().strip()
            if input_type != 'DeliveriesProblemInput':
                raise ValueError(f'Input file `{input_file_name}` is not a deliveries input.')
            try:
                input_name = input_file.readline().strip()
                orders = tuple(PizzaOrder.deserialize(serialized_order, roads)
                               for serialized_order in input_file.readline().split(';'))
                pizza_stores = tuple(PizzaStore.deserialize(serialized_pizza_store, roads)
                                   for serialized_pizza_store in input_file.readline().split(';'))
                deliveries_vehicles = tuple(DeliveriesVehicle.deserialize(serialized_deliveries_vehicle, roads)
                                   for serialized_deliveries_vehicle in input_file.readline().split(';'))
            except:
                raise ValueError(f'Invalid input file `{input_file_name}`.')
        return DeliveriesProblemInput(input_name=input_name, orders=orders, pizza_stores=pizza_stores,
                                      deliveries_vehicles=deliveries_vehicles)

    def store_to_file(self, input_file_name: str):
        with open(Consts.get_data_file_path(input_file_name), 'w') as input_file:
            lines = [
                'DeliveriesProblemInput',
                str(self.input_name.strip()),
                ';'.join(order.serialize() for order in self.orders),
                ';'.join(pizza_store.serialize() for pizza_store in self.pizza_stores),
                ';'.join(deliveries_vehicle.serialize() for deliveries_vehicle in self.deliveries_vehicles),
            ]
            for line in lines:
                input_file.write(line + '\n')

    @staticmethod
    def load_all_inputs(roads: Roads) -> Dict[str, 'DeliveriesProblemInput']:
        """
        Loads all the inputs in the inputs directory.
        :return: list of inputs.
        """
        inputs = {}
        input_file_names = [f for f in os.listdir(Consts.DATA_PATH)
                            if os.path.isfile(os.path.join(Consts.DATA_PATH, f)) and f.split('.')[-1] == 'in']
        for input_file_name in input_file_names:
            try:
                problem_input = DeliveriesProblemInput.load_from_file(input_file_name, roads)
                inputs[problem_input.input_name] = problem_input
            except:
                pass
        return inputs

    @property
    def nr_vehicles(self):
        return len(self.deliveries_vehicles)

    @property
    def nr_orders(self):
        return len(self.orders)

    @property
    def min_nr_deliveries_per_vehicle(self):
        return math.floor(self.nr_orders / self.nr_vehicles)

    @property
    def max_nr_deliveries_per_vehicle(self):
        return math.ceil(self.nr_orders / self.nr_vehicles)


class DeliveriesProblem(GraphProblem):
    """
    An instance of this class represents a strict deliveries problem.
    """

    name = 'Deliveries'

    def __init__(self, problem_input: DeliveriesProblemInput, roads: Roads,
                 map_distance_finder: CachedMapDistanceFinder):
        self.name += '({})'.format(problem_input.input_name)
        initial_state = DeliveriesState(
            vehicles=tuple(DeliveriesVehicleState(
                current_location=vehicle.start_location,
                # fuel_level=vehicle.fuel_tank_initial_level,
                nr_loaded_pizzas=0,
                dropped_so_far=frozenset(),
                # distance_travel_meter=0
            ) for vehicle in problem_input.deliveries_vehicles),
            pizza_stores=tuple(PizzaStoreState(nr_visits=0) for _ in problem_input.pizza_stores)
        )
        super(DeliveriesProblem, self).__init__(initial_state)
        self.problem_input = problem_input
        self.roads = roads
        self.map_distance_finder = map_distance_finder

    def _create_deliveries_vehicle_state_for_next_stop_site(
            self, vehicle: DeliveriesVehicle,
            previous_vehicle_state: DeliveriesVehicleState,
            next_stop_site: Union[PizzaOrder, PizzaStore]) -> Optional[Tuple[DeliveriesVehicleState, float]]:
        if isinstance(next_stop_site, PizzaOrder) \
                and len(previous_vehicle_state.dropped_so_far) >= self.problem_input.max_nr_deliveries_per_vehicle:
            return None
        if next_stop_site == previous_vehicle_state.current_location \
            and isinstance(next_stop_site, PizzaStore) \
            and self.problem_input.min_nr_deliveries_per_vehicle <= len(previous_vehicle_state.dropped_so_far):
            return previous_vehicle_state, 0
        if next_stop_site == previous_vehicle_state.current_location:
            return None
        if isinstance(next_stop_site, PizzaStore) and isinstance(previous_vehicle_state.current_location, PizzaStore):
            return None
        if isinstance(next_stop_site, PizzaOrder) and next_stop_site.nr_pizzas > previous_vehicle_state.nr_loaded_pizzas:
            return None
        current_junction = previous_vehicle_state.get_current_junction()
        if vehicle.vehicle_type == DeliveriesVehicleType.Drone:
            distance = next_stop_site.location.calc_air_distance_from(current_junction)
        else:
            distance = self.map_distance_finder.get_map_cost_between(current_junction, next_stop_site.location)
            if distance is None:
                return None
        # if distance >= previous_vehicle_state.fuel_level:
        #     return None
        # new_fuel_level = previous_vehicle_state.fuel_level - distance
        # assert new_fuel_level > 0
        new_nr_loaded_pizzas = previous_vehicle_state.nr_loaded_pizzas
        if isinstance(next_stop_site, PizzaStore):
            # new_fuel_level = vehicle.fuel_tank_capacity
            new_nr_loaded_pizzas = vehicle.pizzas_capacity
        new_dropped_so_far = set(previous_vehicle_state.dropped_so_far)
        if isinstance(next_stop_site, PizzaOrder):
            new_dropped_so_far.add(next_stop_site)
            new_nr_loaded_pizzas -= next_stop_site.nr_pizzas
        assert new_nr_loaded_pizzas >= 0
        return DeliveriesVehicleState(
            current_location=next_stop_site,
            # fuel_level=new_fuel_level,
            nr_loaded_pizzas=new_nr_loaded_pizzas,
            dropped_so_far=frozenset(new_dropped_so_far),
            # distance_travel_meter=previous_vehicle_state.distance_travel_meter + distance
        ), distance

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[Tuple[GraphProblemState, float]]:
        assert isinstance(state_to_expand, DeliveriesState)

        awaiting_orders = set(self.problem_input.orders) - state_to_expand.dropped_so_far
        # pizza_stores_can_be_visited = set(
        #     pizza_store
        #     for pizza_store, pizza_store_state
        #     in zip(self.problem_input.pizza_stores, state_to_expand.pizza_stores)
        #     if pizza_store_state.nr_visits < pizza_store.max_nr_visits)
        pizza_stores_can_be_visited = set(self.problem_input.pizza_stores)
        all_potential_stop_sites = awaiting_orders | pizza_stores_can_be_visited

        # nr_vehicles_forwhome_pizza_store_visit_is_required = sum(
        #     int(vehicle_state.nr_loaded_pizzas < max(0, self.problem_input.min_nr_deliveries_per_vehicle - len(
        #         vehicle_state.dropped_so_far)))
        #     for vehicle_state in state_to_expand.vehicles
        # )
        # total_nr_pizza_stores_visits = sum(
        #     pizza_store.max_nr_visits - pizza_store_state.nr_visits
        #     for pizza_store, pizza_store_state
        #     in zip(self.problem_input.pizza_stores, state_to_expand.pizza_stores)
        #     if pizza_store_state.nr_visits < pizza_store.max_nr_visits)
        # if total_nr_pizza_stores_visits < nr_vehicles_forwhome_pizza_store_visit_is_required:
        #     return

        for potential_stop_site_per_vehicle in itertools.permutations(all_potential_stop_sites, r=self.problem_input.nr_vehicles):
            new_vehicles_states = []
            new_vehicles_costs = []
            for next_stop_site, vehicle, vehicle_state in zip(potential_stop_site_per_vehicle, self.problem_input.deliveries_vehicles, state_to_expand.vehicles):
                next_vehicle_ret = self._create_deliveries_vehicle_state_for_next_stop_site(
                    vehicle, vehicle_state, next_stop_site)
                if next_vehicle_ret is None:
                    break
                next_vehicle_state, next_vehicle_cost = next_vehicle_ret
                new_vehicles_states.append(next_vehicle_state)
                new_vehicles_costs.append(next_vehicle_cost)
            if len(new_vehicles_states) != self.problem_input.nr_vehicles:
                continue

            # pizza_stores_states = tuple(
            #     PizzaStoreState(
            #         nr_visits=pizza_store_state.nr_visits +
            #                   sum(1 for new_vehicles_state, prev_vehicles_state in zip(new_vehicles_states, state_to_expand.vehicles)
            #                       if new_vehicles_state.current_location == pizza_store and prev_vehicles_state.current_location != pizza_store))
            #     for pizza_store_state, pizza_store in zip(state_to_expand.pizza_stores, self.problem_input.pizza_stores)
            # )
            pizza_stores_states = state_to_expand.pizza_stores

            new_state = DeliveriesState(vehicles=tuple(new_vehicles_states), pizza_stores=pizza_stores_states)
            if new_state == state_to_expand:
                continue

            # TODO: i'm not so sure about the `max_traveled_distance` as the action cost.
            # operator_cost = new_state.max_traveled_distance_so_far - state_to_expand.max_traveled_distance_so_far

            operator_cost = sum(
                next_vehicle_cost  # next_vehicle_state.distance_travel_meter - prev_vehicle_state.distance_travel_meter
                for prev_vehicle_state, next_vehicle_state, next_vehicle_cost in zip(state_to_expand.vehicles, new_vehicles_states, new_vehicles_costs))

            # operator_cost = sum(
            #     vehicle_state.distance_travel_meter if isinstance(vehicle_state.current_location, PizzaOrder) else 0
            #     for vehicle_state in new_vehicles_states)

            assert operator_cost >= 0

            # print('operator_cost', operator_cost)

            yield new_state, operator_cost

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        TODO: implement this method!
        """
        assert isinstance(state, DeliveriesState)

        # raise NotImplemented()  # TODO: remove!

        return state.dropped_so_far == set(self.problem_input.orders) \
               and all(isinstance(vehicle_state.current_location, PizzaStore) for vehicle_state in state.vehicles)

        # if ret:
        #     total_distance_travel = sum(vehicle_state.distance_travel_meter for vehicle_state in state.vehicles)
        #     print(total_distance_travel)
        # return False


class DeliveriesHeuristic(HeuristicFunction):
    heuristic_name = 'MSTMapDist'

    def __init__(self, problem: GraphProblem, map_distance_finder: CachedMapDistanceFinder):
        self.map_distance_finder = map_distance_finder
        super(DeliveriesHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesProblem)

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, DeliveriesProblem)
        assert isinstance(state, DeliveriesState)

        # max_traveled_distance_so_far = state.max_traveled_distance_so_far
        # max_remained_distance = max(
        #     self._calculate_min_remained_distance_for_vehicle(state, vehicle_state, vehicle) +
        #     vehicle_state.distance_travel_meter - max_traveled_distance_so_far
        #     for vehicle_state, vehicle in zip(state.vehicles, self.problem.problem_input.deliveries_vehicles)
        # )
        # return max(max_remained_distance, 0)

        # summed_min_distances_per_vehicle = sum(
        #     self._calculate_min_remained_distance_for_vehicle(state, vehicle_state, vehicle)
        #     for vehicle_state, vehicle in zip(state.vehicles, self.problem.problem_input.deliveries_vehicles)
        # )
        # return summed_min_distances_per_vehicle

        min_remained_distance_by_common_mst = self._calculate_min_remained_distance_by_common_mst(state)
        return min_remained_distance_by_common_mst

        # print('heuristic', summed_min_distances_per_vehicle, min_remained_distance_by_common_mst)

        # return max(summed_min_distances_per_vehicle, min_remained_distance_by_common_mst)

    def _calculate_min_remained_distance_by_common_mst(
            self, deliveries_state: DeliveriesState):
        awaiting_orders = set(self.problem.problem_input.orders) - deliveries_state.dropped_so_far
        remained_drop_points = {order.location for order in awaiting_orders}
        current_active_vehicles_states = {vehicle_state for vehicle_state in deliveries_state.vehicles
            if not isinstance(vehicle_state.current_location, PizzaStore)
               or len(vehicle_state.dropped_so_far) < self.problem.problem_input.min_nr_deliveries_per_vehicle}
        # current_vehicles_locations = {
        #     vehicle_state.get_current_junction() for vehicle_state in current_active_vehicles_states}
        pizza_stores_locations = {pizza_store.location for pizza_store in self.problem.problem_input.pizza_stores}
        # all_junctions_to_visit = current_vehicles_locations | remained_drop_points  # | pizza_stores_locations
        # min_nr_hops_to_final_station = sum(
        #     int(not isinstance(vehicle_state.current_location, PizzaStore)
        #         or len(vehicle_state.dropped_so_far) < self.problem.problem_input.min_nr_deliveries_per_vehicle)
        #     for vehicle_state in deliveries_state.vehicles
        # )

        # min_nr_hops_left = len(remained_drop_points)  # + min_nr_hops_to_final_station
        # min_nr_hops_left = None
        # minimum_cost_of_remaining_hops = self._calculate_junctions_dist_mst_weight(
        #     all_junctions_to_visit, 'map', min_nr_hops_left)

        is_pizza_store_visit_required = any(
            vehicle_state.nr_loaded_pizzas < max(0, self.problem.problem_input.min_nr_deliveries_per_vehicle - len(vehicle_state.dropped_so_far))
            for vehicle_state in deliveries_state.vehicles
        )

        # pizza_stores_can_be_visited = set(
        #     pizza_store
        #     for pizza_store, pizza_store_state
        #     in zip(self.problem.problem_input.pizza_stores, deliveries_state.pizza_stores)
        #     if pizza_store_state.nr_visits < pizza_store.max_nr_visits)
        pizza_stores_can_be_visited = self.problem.problem_input.pizza_stores

        # if is_pizza_store_visit_required and len(pizza_stores_can_be_visited) < 1:
        #     return np.inf

        all_remaining_sites = awaiting_orders | {vehicle_state.current_location for vehicle_state in current_active_vehicles_states}
        class DummyRootJunction:
            pass
        all_remaining_sites.add(DummyRootJunction())
        def distance_between_sites(site1: Union[DummyRootJunction, Junction, PizzaOrder, PizzaStore], site2: Union[DummyRootJunction, Junction, PizzaOrder, PizzaStore]) -> float:
            sites = (site1, site2)
            one_is_root = any(isinstance(site, DummyRootJunction) for site in sites)
            # one_is_pizza_store = any(isinstance(site, PizzaStore) for site in sites)
            one_is_vehicle = any(
                site == vehicle_state.current_location for vehicle_state in deliveries_state.vehicles for site in sites)
            vehicle_to_vehicle = all(
                any(site == vehicle_state.current_location for vehicle_state in deliveries_state.vehicles)
                for site in sites)
            one_is_awaiting_orders = any(site in awaiting_orders for site in sites)
            assert 1 <= int(one_is_root)+int(one_is_vehicle)+int(one_is_awaiting_orders) <= 2
            if one_is_root:
                return 1 if one_is_vehicle else np.inf
            elif vehicle_to_vehicle:
                return np.inf
            junctions = tuple(site if isinstance(site, Junction) else site.location for site in sites)
            return self._get_symmetric_distance_between_junctions(*junctions, distance_type='map')
        if is_pizza_store_visit_required:
            minimum_cost_of_remaining_hops = min(
                self._calculate_mst_weight(
                    all_remaining_sites | {pizza_store}, distance_between_sites)
                for pizza_store in pizza_stores_can_be_visited
            )
        else:
            minimum_cost_of_remaining_hops = self._calculate_mst_weight(
                all_remaining_sites, distance_between_sites)
        minimum_cost_of_remaining_hops -= len(current_active_vehicles_states)

        min_nr_hops_to_final_station_when_not_at_very_end = sum(
            int(len(vehicle_state.dropped_so_far) < self.problem.problem_input.min_nr_deliveries_per_vehicle)
            for vehicle_state in deliveries_state.vehicles
        )
        minimum_cost_of_final_hops = 0 if len(remained_drop_points) == 0 else min_nr_hops_to_final_station_when_not_at_very_end * min(
            self._get_symmetric_distance_between_junctions(junc_to_visit, pizza_store_location, 'map')
            for junc_to_visit in remained_drop_points for pizza_store_location in pizza_stores_locations)

        minimum_cost_of_final_hops += sum(
            min(
                self._get_symmetric_distance_between_junctions(vehicle_state.get_current_junction(), pizza_store_location, 'map')
                for pizza_store_location in pizza_stores_locations)
            for vehicle_state in deliveries_state.vehicles
            if not isinstance(vehicle_state.current_location, PizzaStore)
            and len(vehicle_state.dropped_so_far) >= self.problem.problem_input.min_nr_deliveries_per_vehicle
        )

        # print(f'awaiting_orders: {len(awaiting_orders)} -- \tmin_nr_hops_left: {min_nr_hops_left} -- \tmin_nr_hops_to_final_station: {min_nr_hops_to_final_station} -- \tminimum_cost_of_remaining_hops: {minimum_cost_of_remaining_hops}')
        return minimum_cost_of_remaining_hops + minimum_cost_of_final_hops

    def _calculate_min_remained_distance_for_vehicle(
            self, deliveries_state: DeliveriesState, vehicle_state: DeliveriesVehicleState, vehicle: DeliveriesVehicle):
        awaiting_orders = set(self.problem.problem_input.orders) - deliveries_state.dropped_so_far
        remained_drop_points = {order.location for order in awaiting_orders}
        all_junctions_to_visit = remained_drop_points | {vehicle_state.get_current_junction()}
        distance_type = 'air' if vehicle.vehicle_type == DeliveriesVehicleType.Drone else 'map'
        min_nr_junctions_must_visit = max(0, self.problem.problem_input.min_nr_deliveries_per_vehicle - \
                                      len(vehicle_state.dropped_so_far))

        # if not isinstance(vehicle_state.current_location, PizzaStore) or min_nr_junctions_must_visit > 0:
        #     min_nr_junctions_must_visit += 1
        #     all_junctions_to_visit |= {pizza_store.location for pizza_store in self.problem.problem_input.pizza_stores}

        if min_nr_junctions_must_visit == 0:
            if isinstance(vehicle_state.current_location, PizzaStore):
                return 0
            minimum_cost_of_final_hop_to_pizza_store = min(
                self._get_symmetric_distance_between_junctions(
                    vehicle_state.current_location.location, pizza_store.location, distance_type)
                for pizza_store in self.problem.problem_input.pizza_stores
            )
            return minimum_cost_of_final_hop_to_pizza_store

        minimum_cost_of_final_hop_to_pizza_store = min(
            self._get_symmetric_distance_between_junctions(
                awaiting_order.location, pizza_store.location, distance_type)
            for pizza_store in self.problem.problem_input.pizza_stores
            for awaiting_order in awaiting_orders
        )

        # min_nr_junctions_must_visit = None  # TODO: remove!
        minimum_cost_of_remaining_drop_points = self._calculate_junctions_dist_mst_weight(
            all_junctions_to_visit, distance_type, min_nr_junctions_must_visit)

        return minimum_cost_of_remaining_drop_points + minimum_cost_of_final_hop_to_pizza_store

    def _get_symmetric_distance_between_junctions(
            self, junction1: Junction, junction2: Junction, distance_type: str):
        assert distance_type in {'air', 'map'}
        if distance_type == 'air':
            return junction1.calc_air_distance_from(junction2)
        else:
            # FIXME: problem - this min(->, <-) calculation breaks the triangle inequality of the space.
            #  Because sum[(1)->(2)<-(3)] could be < than sum[(1)->(3)]
            #  Solution: use here a SymmetricMapProblem in which
            #   (1) all routes bi-directional; and
            #   (2) same speed for each route lane direction.
            dist12 = self.map_distance_finder.get_map_cost_between(junction1, junction2)
            dist21 = self.map_distance_finder.get_map_cost_between(junction2, junction1)
            distances = list(filter(lambda d: d is not None, [dist12, dist21]))
            assert len(distances) > 0
            return min(distances)

    def _calculate_junctions_dist_mst_weight(
            self, junctions: Set[Junction], distance_type: str, min_nr_junctions_must_visit: Optional[int] = None) -> float:
        assert distance_type in {'air', 'map'}
        def junctions_dist_fn(junction1: Junction, junction2: Junction):
            return self._get_symmetric_distance_between_junctions(junction1, junction2, distance_type)
        return self._calculate_mst_weight(junctions, junctions_dist_fn, min_nr_junctions_must_visit)

    def _alloc_matrix_for_mst(self, n: int) -> np.ndarray:
        if not hasattr(self, '_mst_matrix_dict'):
            self._mst_matrix_dict = {}
        if n not in self._mst_matrix_dict:
            self._mst_matrix_dict[n] = np.zeros((n, n), dtype=np.float)
        return self._mst_matrix_dict[n]

    def _calculate_mst_weight(
            self, vertices: Set[object], edges_costs_fn, at_least_k_vertices: Optional[int] = None) -> float:
        nr_junctions = len(vertices)
        idx_to_junction = {idx: junction for idx, junction in enumerate(vertices)}
        distances_matrix = np.zeros((nr_junctions, nr_junctions), dtype=np.float)
        # distances_matrix = self._alloc_matrix_for_mst(nr_junctions)
        for j1_idx in range(nr_junctions):
            for j2_idx in range(nr_junctions):
                if j1_idx == j2_idx:
                    continue
                dist = edges_costs_fn(idx_to_junction[j1_idx], idx_to_junction[j2_idx])
                distances_matrix[j1_idx, j2_idx] = dist
                distances_matrix[j2_idx, j1_idx] = dist
        distances_mst = mst(distances_matrix)

        if at_least_k_vertices is None:
            return float(np.sum(distances_mst))

        # Get the total cost sum of #`min_nr_junctions_must_visit` least weighted edges from `distances_mst`.
        nonzero_mst_edges_weights = np.asarray(distances_mst[np.nonzero(distances_mst)]).ravel()
        nonzero_mst_edges_weights_sorted = np.sort(nonzero_mst_edges_weights)

        nr_edges = nonzero_mst_edges_weights_sorted.size
        least_weighted_edges = nonzero_mst_edges_weights_sorted[0:(at_least_k_vertices % nr_edges)]
        return (at_least_k_vertices // nr_edges) * np.sum(nonzero_mst_edges_weights_sorted) + np.sum(least_weighted_edges)
