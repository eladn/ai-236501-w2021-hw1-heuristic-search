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
    loaded_deliveries: FrozenSet[Delivery]
    dropped_deliveries: FrozenSet[Delivery]
    current_location: Junction

    def get_last_action(self) -> Optional[Tuple[str, Delivery]]:
        found_loaded_delivery = next((loaded_delivery for loaded_delivery in self.loaded_deliveries
                                      if loaded_delivery.pick_location == self.current_location), None)
        if found_loaded_delivery is not None:
            return 'pick', found_loaded_delivery

        found_dropped_delivery = next((dropped_delivery for dropped_delivery in self.dropped_deliveries
                                       if dropped_delivery.drop_location == self.current_location), None)
        if found_dropped_delivery is not None:
            return 'drop', found_dropped_delivery

        return None

    def __str__(self):
        last_action = self.get_last_action()
        current_location = 'initial-location' if last_action is None else f'{last_action[0]} loc @ {last_action[1]}'

        return f'(dropped: {list(self.dropped_deliveries)} ' \
               f'loaded: {list(self.loaded_deliveries)} ' \
               f'current_location: {current_location})'

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
    optimization_objective: OptimizationObjective = OptimizationObjective.Distance

    def __add__(self, other):
        assert isinstance(other, DeliveryCost)
        assert other.optimization_objective == self.optimization_objective
        return DeliveryCost(optimization_objective=self.optimization_objective,
                            distance_cost=self.distance_cost + other.distance_cost,
                            time_cost=self.time_cost + other.time_cost,
                            money_cost=self.money_cost + other.money_cost)

    def get_g_cost(self) -> float:
        if self.optimization_objective == OptimizationObjective.Distance:
            return self.distance_cost
        elif self.optimization_objective == OptimizationObjective.Time:
            return self.time_cost
        else:
            assert self.optimization_objective == OptimizationObjective.Money
            return self.money_cost

    def __repr__(self):
        return f'DeliveryCost(' \
               f'dist={self.distance_cost:11.3f} meter, ' \
               f'time={self.time_cost:11.3f} minutes, ' \
               f'money={self.money_cost:11.3f} nis)'


class DeliveriesTruckProblem(GraphProblem):
    def __init__(self,
                 problem_input: DeliveriesTruckProblemInput,
                 roads: Roads,
                 optimization_objective: OptimizationObjective = OptimizationObjective.Distance):
        self.name += '({})'.format(problem_input.input_name)
        initial_state = DeliveriesTruckState(
            loaded_deliveries=frozenset(),
            dropped_deliveries=frozenset(),
            current_location=problem_input.delivery_truck.initial_location
        )
        super(DeliveriesTruckProblem, self).__init__(initial_state)
        self.problem_input = problem_input
        self.roads = roads
        inner_map_problem_heuristic_type = lambda problem: TruckDeliveriesInnerMapProblemHeuristic(problem, self)
        self.map_distance_finder = CachedMapDistanceFinder(
            roads, AStar(inner_map_problem_heuristic_type),
            road_cost_fn=self._calc_map_road_cost,
            zero_road_cost=DeliveryCost(optimization_objective=optimization_objective))
        self.optimization_objective = optimization_objective

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
            assert operator_cost.optimization_objective == self.optimization_objective
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
            assert operator_cost.optimization_objective == self.optimization_objective
            yield OperatorResult(
                successor_state=new_state,
                operator_cost=operator_cost,
                operator_name=f'drop {delivery.client_name}')

    def _calc_map_road_cost(self, link: Link) -> DeliveryCost:
        optimal_velocity, gas_cost_per_meter = self.problem_input.delivery_truck.calc_optimal_driving_parameters(
            self.optimization_objective, max_driving_speed=link.max_speed)
        return DeliveryCost(
            distance_cost=link.distance,
            time_cost=link.distance / optimal_velocity,
            money_cost=gas_cost_per_meter * link.distance + (self.problem_input.toll_road_cost_per_meter * link.distance if link.is_toll_road else 0),
            optimization_objective=self.optimization_objective)

    def is_goal(self, state: GraphProblemState) -> bool:
        assert isinstance(state, DeliveriesTruckState)
        return state.dropped_deliveries == set(self.problem_input.deliveries)

    def get_zero_cost(self) -> Cost:
        return DeliveryCost(optimization_objective=self.optimization_objective)

    def get_cost_lower_bound_from_distance_lower_bound(self, total_distance_lower_bound: float) -> float:
        if self.optimization_objective == OptimizationObjective.Distance:
            return total_distance_lower_bound
        elif self.optimization_objective == OptimizationObjective.Time:
            return total_distance_lower_bound / MAX_ROAD_SPEED
        else:
            assert self.optimization_objective == OptimizationObjective.Money

            lower_bound_for_gas_cost_of_driving_remaining_roads = self._calc_map_road_cost(
                Link(0, 0, total_distance_lower_bound, 0, None, MAX_ROAD_SPEED, False)).money_cost  # TODO: make it better!
            return lower_bound_for_gas_cost_of_driving_remaining_roads


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
        total_distance_lower_bound = self._calculate_junctions_mst_weight_using_air_distance(all_junctions_to_visit)
        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)

    def _calculate_junctions_mst_weight_using_air_distance(
            self, junctions: Set[Junction]) -> float:
        def junctions_dist_fn(junction1: Junction, junction2: Junction) -> float:
            return junction1.calc_air_distance_from(junction2)
        return self._calculate_mst_weight(vertices=junctions, edges_costs_fn=junctions_dist_fn)

    def _calculate_mst_weight(
            self, vertices: Set[object], edges_costs_fn: Callable) -> float:
        nr_junctions = len(vertices)
        idx_to_junction = {idx: junction for idx, junction in enumerate(vertices)}
        distances_matrix = np.zeros((nr_junctions, nr_junctions), dtype=np.float)

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
