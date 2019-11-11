from typing import *
from dataclasses import dataclass
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree as mst

from framework import *
from .map_problem import MapProblem, MapState
from .cached_map_distance_finder import CachedMapDistanceFinder
from .deliveries_problem_input import *


@dataclass(frozen=True)
class DeliveriesTruckState(GraphProblemState):
    loaded_deliveries: FrozenSet[PackagesDelivery]
    dropped_deliveries: FrozenSet[PackagesDelivery]
    current_location: Junction

    def __str__(self):
        return f'dropped: {self.dropped_deliveries} -- loaded: {self.loaded_deliveries} -- current_location: {self.current_location.index}'

    def __eq__(self, other):
        assert isinstance(other, DeliveriesTruckState)
        return self.loaded_deliveries == other.loaded_deliveries \
               and self.dropped_deliveries == other.dropped_deliveries \
               and self.current_location == other.current_location

    def __hash__(self):
        return hash((self.loaded_deliveries, self.dropped_deliveries, self.current_location))


@dataclass(frozen=True)
class DeliveryCost(ExtendedCost):
    distance_cost: float = 0.0
    time_cost: float = 0.0
    money_cost: float = 0.0
    target_type: OptimizationTargetType = OptimizationTargetType.Distance

    def __add__(self, other):
        assert isinstance(other, DeliveryCost)
        assert other.target_type == self.target_type
        return DeliveryCost(target_type=self.target_type,
                            distance_cost=self.distance_cost + other.distance_cost,
                            time_cost=self.time_cost + other.time_cost,
                            money_cost=self.money_cost + other.money_cost)

    def get_g_cost(self) -> float:
        if self.target_type == OptimizationTargetType.Distance:
            return self.distance_cost
        elif self.target_type == OptimizationTargetType.Time:
            return self.time_cost
        else:
            assert self.target_type == OptimizationTargetType.Money
            return self.money_cost


class DeliveriesTruckProblem(GraphProblem):
    def __init__(self,
                 problem_input: DeliveriesTruckProblemInput,
                 roads: Roads,
                 target_type: OptimizationTargetType = OptimizationTargetType.Distance):
        self.name += '({})'.format(problem_input.input_name)
        initial_state = DeliveriesTruckState(
            loaded_deliveries=frozenset(),
            dropped_deliveries=frozenset(),
            current_location=problem_input.delivery_truck.initial_location
        )
        super(DeliveriesTruckProblem, self).__init__(initial_state)
        self.problem_input = problem_input
        self.roads = roads
        # self.solver_for_inner_map_problem = solver_for_inner_map_problem
        inner_map_problem_heuristic_type = lambda problem: TruckDeliveriesInnerMapProblemHeuristic(problem, self)
        self.map_distance_finder = CachedMapDistanceFinder(
            roads, AStar(inner_map_problem_heuristic_type),
            road_cost_fn=self._calc_map_road_cost,
            zero_road_cost=DeliveryCost(target_type=target_type))
        self.target_type = target_type

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[OperatorResult]:
        assert isinstance(state_to_expand, DeliveriesTruckState)

        # Pick delivery
        deliveries_waiting_to_pick = (set(self.problem_input.deliveries) - state_to_expand.dropped_deliveries) - state_to_expand.loaded_deliveries
        free_space_in_truck = self.problem_input.delivery_truck.max_nr_loaded_packages - len(state_to_expand.loaded_deliveries)
        for delivery in deliveries_waiting_to_pick:
            if delivery.nr_packages > free_space_in_truck:
                continue
            new_state = DeliveriesTruckState(
                loaded_deliveries=frozenset(state_to_expand.loaded_deliveries | {delivery}),
                dropped_deliveries=state_to_expand.dropped_deliveries,
                current_location=delivery.pick_location
            )
            operator_cost = self.map_distance_finder.get_map_cost_between(
                state_to_expand.current_location, delivery.pick_location)
            assert isinstance(operator_cost, DeliveryCost)
            assert operator_cost.target_type == self.target_type
            # operator_cost += DeliveryCost(
            #     target_type=self.target_type,
            #     money_cost=self.problem_input.delivery_truck.gas_cost_addition_per_meter_per_loaded_package *
            #                sum(delivery.nr_packages for delivery in state_to_expand.loaded_deliveries) *
            #                operator_cost.distance_cost)
            yield OperatorResult(
                successor_state=new_state,
                operator_cost=operator_cost,
                operator_name=f'pick {delivery.client_name}')

        # Drop delivery
        for delivery in state_to_expand.loaded_deliveries:
            new_state = DeliveriesTruckState(
                loaded_deliveries=frozenset(state_to_expand.loaded_deliveries - {delivery}),
                dropped_deliveries=frozenset(state_to_expand.dropped_deliveries | {delivery}),
                current_location=delivery.drop_location
            )
            operator_cost = self.map_distance_finder.get_map_cost_between(
                state_to_expand.current_location, delivery.drop_location)
            assert isinstance(operator_cost, DeliveryCost)
            assert operator_cost.target_type == self.target_type
            # operator_cost += DeliveryCost(
            #     target_type=self.target_type,
            #     money_cost=self.problem_input.delivery_truck.gas_cost_addition_per_meter_per_loaded_package *
            #                sum(delivery.nr_packages for delivery in state_to_expand.loaded_deliveries) *
            #                operator_cost.distance_cost)
            yield OperatorResult(
                successor_state=new_state,
                operator_cost=operator_cost,
                operator_name=f'drop {delivery.client_name}')

    def _calc_map_road_cost(self, link: Link) -> DeliveryCost:
        optimal_velocity, gas_cost_per_meter = self.problem_input.delivery_truck.calc_optimal_driving_parameters(
            self.target_type, max_driving_speed=link.max_speed)
        return DeliveryCost(
            distance_cost=link.distance,
            time_cost=link.distance / optimal_velocity,
            money_cost=gas_cost_per_meter * link.distance + (self.problem_input.toll_road_cost_per_meter * link.distance if link.is_toll_road else 0),
            target_type=self.target_type)

    def is_goal(self, state: GraphProblemState) -> bool:
        assert isinstance(state, DeliveriesTruckState)
        return state.dropped_deliveries == set(self.problem_input.deliveries)

    def get_zero_cost(self) -> Cost:
        return DeliveryCost(target_type=self.target_type)

    def get_cost_lower_bound_from_distance_lower_bound(self, total_distance_lower_bound: float) -> float:
        if self.target_type == OptimizationTargetType.Distance:
            return total_distance_lower_bound
        elif self.target_type == OptimizationTargetType.Time:
            return total_distance_lower_bound / MAX_ROAD_SPEED
        else:
            assert self.target_type == OptimizationTargetType.Money

            lower_bound_for_gas_cost_of_driving_remaining_roads = self._calc_map_road_cost(Link(0, 0, total_distance_lower_bound, 0, None, MAX_ROAD_SPEED, False)).money_cost  # TODO: make it better!

            # lower_bound_for_carrying_distance_times_nr_packages = sum(
            #     self._get_symmetric_distance_between_junctions(
            #         state.current_location, delivery.drop_location, 'air') * delivery.nr_packages
            #     for delivery in state.loaded_deliveries
            # ) + sum(
            #     self._get_symmetric_distance_between_junctions(
            #         delivery.pick_location, delivery.drop_location, 'air') * delivery.nr_packages
            #     for delivery in deliveries_waiting_to_pick
            # )
            # additional_gas_cost_for_carrying_loaded_packages = \
            #     self.problem_input.delivery_truck.gas_cost_addition_per_meter_per_loaded_package * \
            #     lower_bound_for_carrying_distance_times_nr_packages

            return lower_bound_for_gas_cost_of_driving_remaining_roads  # + additional_gas_cost_for_carrying_loaded_packages


class TruckDeliveriesHeuristic(HeuristicFunction):
    heuristic_name = 'MSTAirDistTruckDeliveries'

    def __init__(self, problem: GraphProblem):
        super(TruckDeliveriesHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesTruckProblem)

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, DeliveriesTruckProblem)
        assert isinstance(state, DeliveriesTruckState)

        deliveries_waiting_to_pick = (set(
            self.problem.problem_input.deliveries) - state.dropped_deliveries) - state.loaded_deliveries
        all_junctions_to_visit = {delivery.pick_location for delivery in deliveries_waiting_to_pick} | \
                                 {delivery.drop_location for delivery in state.loaded_deliveries} | \
                                 {state.current_location}
        total_distance_lower_bound = self._calculate_junctions_dist_mst_weight(all_junctions_to_visit, 'air')
        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)

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
            #   (2) same speed&length for each route lane direction.
            dist12 = self.map_distance_finder.get_map_cost_between(junction1, junction2)
            dist21 = self.map_distance_finder.get_map_cost_between(junction2, junction1)
            distances = list(filter(lambda d: d is not None, [dist12, dist21]))
            assert len(distances) > 0
            return min(distances)

    def _calculate_junctions_dist_mst_weight(
            self, junctions: Set[Junction], distance_type: str) -> float:
        assert distance_type in {'air', 'map'}
        def junctions_dist_fn(junction1: Junction, junction2: Junction):
            return self._get_symmetric_distance_between_junctions(junction1, junction2, distance_type)
        return self._calculate_mst_weight(junctions, junctions_dist_fn)

    def _calculate_mst_weight(
            self, vertices: Set[object], edges_costs_fn) -> float:
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
        return float(np.sum(distances_mst))


class TruckDeliveriesInnerMapProblemHeuristic(HeuristicFunction):
    heuristic_name = 'AirDist'

    def __init__(self, inner_map_problem: GraphProblem, outer_deliveries_problem: DeliveriesTruckProblem):
        super(TruckDeliveriesInnerMapProblemHeuristic, self).__init__(inner_map_problem)
        assert isinstance(self.problem, MapProblem)
        self.outer_deliveries_problem = outer_deliveries_problem

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, MapProblem)
        assert isinstance(state, MapState)

        source_junction = self.problem.roads[state.junction_id]
        target_junction = self.problem.roads[self.problem.target_junction_id]
        total_distance_lower_bound = source_junction.calc_air_distance_from(target_junction)
        return self.outer_deliveries_problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)