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

        # raise NotImplemented()  # TODO: remove!

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
    name = 'Deliveries'

    def __init__(self,
                 problem_input: DeliveriesTruckProblemInput,
                 streets_map: StreetsMap,
                 optimization_objective: OptimizationObjective = OptimizationObjective.Distance):
        self.name += f'({problem_input.input_name}({len(problem_input.deliveries)}):{optimization_objective.name})'
        initial_state = DeliveriesTruckState(
            loaded_deliveries=frozenset(),
            dropped_deliveries=frozenset(),
            current_location=problem_input.delivery_truck.initial_location
        )
        super(DeliveriesTruckProblem, self).__init__(initial_state)
        self.problem_input = problem_input
        self.streets_map = streets_map
        inner_map_problem_heuristic_type = lambda problem: TruckDeliveriesInnerMapProblemHeuristic(problem, self)
        self.map_distance_finder = CachedMapDistanceFinder(
            streets_map, AStar(inner_map_problem_heuristic_type),
            road_cost_fn=self._calc_map_road_cost,
            zero_road_cost=DeliveryCost(optimization_objective=optimization_objective))
        self.optimization_objective = optimization_objective

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[OperatorResult]:
        assert isinstance(state_to_expand, DeliveriesTruckState)

        """
        TODO: 
        """
        # raise NotImplemented()  # TODO: remove!

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
        return DeliveryCost(
            distance_cost=link.distance,
            time_cost=0,
            money_cost=0,
            optimization_objective=self.optimization_objective)  # TODO: modify!

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
            # raise NotImplemented()  # TODO: remove!
            return total_distance_lower_bound / MAX_ROAD_SPEED
        else:
            assert self.optimization_objective == OptimizationObjective.Money
            # raise NotImplemented()  # TODO: remove!
            lower_bound_for_gas_cost_of_driving_remaining_roads = self._calc_map_road_cost(
                Link(0, 0, total_distance_lower_bound, 0, MAX_ROAD_SPEED, False)).money_cost
            return lower_bound_for_gas_cost_of_driving_remaining_roads


class TruckDeliveriesMaxAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'TruckDeliveriesMaxAirDist'

    def __init__(self, problem: GraphProblem):
        super(TruckDeliveriesMaxAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesTruckProblem)

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, DeliveriesTruckProblem)
        assert isinstance(state, DeliveriesTruckState)

        # raise NotImplemented()  # TODO: remove!

        deliveries_waiting_to_pick = (set(
            self.problem.problem_input.deliveries) - state.dropped_deliveries) - state.loaded_deliveries
        all_junctions_to_visit = {delivery.pick_location for delivery in deliveries_waiting_to_pick} | \
                                 {delivery.drop_location for delivery in deliveries_waiting_to_pick} | \
                                 {delivery.drop_location for delivery in state.loaded_deliveries} | \
                                 {state.current_location}
        if len(all_junctions_to_visit) < 2:
            return 0
        total_distance_lower_bound = max(
            junction1.calc_air_distance_from(junction2)
            for junction1 in all_junctions_to_visit
            for junction2 in all_junctions_to_visit
            if junction1 != junction2)
        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)


class TruckDeliveriesSumAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'TruckDeliveriesSumAirDist'

    def __init__(self, problem: GraphProblem):
        super(TruckDeliveriesSumAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesTruckProblem)
        self.junctions_pair_to_air_distances_mapping = {}

    def get_distance_between_junctions(self, junction1: Junction, junction2: Junction) -> float:
        key = frozenset((junction1, junction2))
        if key not in self.junctions_pair_to_air_distances_mapping:
            self.junctions_pair_to_air_distances_mapping[key] = junction1.calc_air_distance_from(junction2)
        return self.junctions_pair_to_air_distances_mapping[key]

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, DeliveriesTruckProblem)
        assert isinstance(state, DeliveriesTruckState)

        # raise NotImplemented()  # TODO: remove!

        deliveries_waiting_to_pick = (set(
            self.problem.problem_input.deliveries) - state.dropped_deliveries) - state.loaded_deliveries
        all_junctions_to_visit = {delivery.pick_location for delivery in deliveries_waiting_to_pick} | \
                                 {delivery.drop_location for delivery in deliveries_waiting_to_pick} | \
                                 {delivery.drop_location for delivery in state.loaded_deliveries} | \
                                 {state.current_location}
        if len(all_junctions_to_visit) < 2:
            return 0

        last_location = state.current_location
        total_distance_sum = 0
        while len(all_junctions_to_visit) > 1:
            all_junctions_to_visit.remove(last_location)
            locs_and_dist = [(loc, self.get_distance_between_junctions(last_location, loc)) for loc in all_junctions_to_visit]
            min_dist_idx = np.argmin(np.array([dist for _, dist in locs_and_dist]))
            next_location = locs_and_dist[min_dist_idx][0]
            total_distance_sum += self.get_distance_between_junctions(last_location, next_location)
            last_location = next_location

        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_sum)


class TruckDeliveriesMSTAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'TruckDeliveriesMSTAirDist'

    def __init__(self, problem: GraphProblem):
        super(TruckDeliveriesMSTAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesTruckProblem)

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, DeliveriesTruckProblem)
        assert isinstance(state, DeliveriesTruckState)

        """
        TODO:
        
        """
        # raise NotImplemented()  # TODO: remove!

        deliveries_waiting_to_pick = (set(
            self.problem.problem_input.deliveries) - state.dropped_deliveries) - state.loaded_deliveries
        all_junctions_to_visit = {delivery.pick_location for delivery in deliveries_waiting_to_pick} | \
                                 {delivery.drop_location for delivery in deliveries_waiting_to_pick} | \
                                 {delivery.drop_location for delivery in state.loaded_deliveries} | \
                                 {state.current_location}
        total_distance_lower_bound = self._calculate_junctions_mst_weight_using_air_distance(all_junctions_to_visit)
        # print(
        #     f'#deliveries: {len(self.problem.problem_input.deliveries)} -- '
        #     f'#loaded: {len(state.loaded_deliveries)} -- '
        #     f'#dropped: {len(state.dropped_deliveries)} -- '
        #     f'#deliveries_waiting_to_pick: {len(deliveries_waiting_to_pick)} -- '
        #     f'#all_junctions_to_visit: {len(all_junctions_to_visit)} --'
        #     f'total_distance_lower_bound: {total_distance_lower_bound}')
        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)

    def _calculate_junctions_mst_weight_using_air_distance(
            self, junctions: Set[Junction]) -> float:
        def junctions_dist_fn(junction1: Junction, junction2: Junction) -> float:
            return junction1.calc_air_distance_from(junction2)
        return self._calculate_mst_weight(vertices=junctions, edges_costs_fn=junctions_dist_fn)

    def _calculate_mst_weight(
            self, vertices: Set[object], edges_costs_fn: Callable) -> float:
        nr_vertices = len(vertices)
        idx_to_vertex = {idx: vertex for idx, vertex in enumerate(vertices)}
        edges_costs_matrix = np.zeros((nr_vertices, nr_vertices), dtype=np.float)

        """
        TODO:
        """
        # raise NotImplemented()  # TODO: remove!

        for j1_idx in range(nr_vertices):
            for j2_idx in range(nr_vertices):
                if j1_idx == j2_idx:
                    continue
                edge_cost = edges_costs_fn(idx_to_vertex[j1_idx], idx_to_vertex[j2_idx])
                edges_costs_matrix[j1_idx, j2_idx] = edge_cost
                edges_costs_matrix[j2_idx, j1_idx] = edge_cost
        mst_matrix = mst(edges_costs_matrix)
        return float(np.sum(mst_matrix))


class TruckDeliveriesInnerMapProblemHeuristic(HeuristicFunction):
    heuristic_name = 'AirDist'

    def __init__(self, inner_map_problem: GraphProblem, outer_deliveries_problem: DeliveriesTruckProblem):
        super(TruckDeliveriesInnerMapProblemHeuristic, self).__init__(inner_map_problem)
        assert isinstance(self.problem, MapProblem)
        self.outer_deliveries_problem = outer_deliveries_problem

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, MapProblem)
        assert isinstance(state, MapState)

        """
        TODO:
        """
        # raise NotImplemented()  # TODO: remove!

        source_junction = self.problem.streets_map[state.junction_id]
        target_junction = self.problem.streets_map[self.problem.target_junction_id]
        total_distance_lower_bound = source_junction.calc_air_distance_from(target_junction)
        return self.outer_deliveries_problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)
