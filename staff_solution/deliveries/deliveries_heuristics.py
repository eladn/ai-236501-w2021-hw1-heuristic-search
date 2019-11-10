from .relaxed_deliveries_problem import RelaxedDeliveriesState, RelaxedDeliveriesProblem
from .strict_deliveries_problem import StrictDeliveriesState, StrictDeliveriesProblem
from .deliveries_problem_input import DeliveriesProblemInput
from framework import *

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from typing import Set, Dict, FrozenSet


class MaxAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MaxAirDist'

    def estimate(self, state: GraphProblemState) -> float:
        """
        Calculates the maximum among air distances between the location
         represented by `state` and the locations of the waiting deliveries.
        TODO: implement this method!
        """
        assert isinstance(self.problem, RelaxedDeliveriesProblem)
        assert isinstance(state, RelaxedDeliveriesState)

        # raise NotImplemented()  # TODO: remove!

        current_location = state.current_location
        dropped_so_far = state.dropped_so_far
        drop_points = self.problem.drop_points
        waiting_deliveries = drop_points - dropped_so_far
        if len(waiting_deliveries) < 1:
            return 0
        return max(
            current_location.calc_air_distance_from(next_junction)
            for next_junction in waiting_deliveries
        )


class MSTAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MSTAirDist'

    def __init__(self, problem: GraphProblem):
        super(MSTAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, RelaxedDeliveriesProblem)
        self._junctions_distances_cache: Dict[FrozenSet[Junction], float] = dict()

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, RelaxedDeliveriesProblem)
        assert isinstance(state, RelaxedDeliveriesState)

        remained_drop_points = set(self.problem.drop_points - state.dropped_so_far)
        remained_drop_points.add(state.current_location)
        return self._calculate_junctions_air_dist_mst_weight(remained_drop_points)

    def _get_distance_between_junctions(self, junction1: Junction, junction2: Junction):
        junctions_pair = frozenset({junction1, junction2})
        if junctions_pair in self._junctions_distances_cache:
            return self._junctions_distances_cache[junctions_pair]
        dist = junction1.calc_air_distance_from(junction2)
        self._junctions_distances_cache[junctions_pair] = dist
        return dist

    def _calculate_junctions_air_dist_mst_weight(self, junctions: Set[Junction]) -> float:
        nr_junctions = len(junctions)
        idx_to_junction = {idx: junction for idx, junction in enumerate(junctions)}
        distances_matrix = np.zeros((nr_junctions, nr_junctions), dtype=np.float)
        for j1_idx in range(nr_junctions):
            for j2_idx in range(nr_junctions):
                if j1_idx == j2_idx:
                    continue
                dist = self._get_distance_between_junctions(idx_to_junction[j1_idx], idx_to_junction[j2_idx])
                distances_matrix[j1_idx, j2_idx] = dist
                distances_matrix[j2_idx, j1_idx] = dist
        return mst(distances_matrix).sum()


class RelaxedDeliveriesHeuristic(HeuristicFunction):
    heuristic_name = 'RelaxedProb'

    def estimate(self, state: GraphProblemState) -> float:
        """
        Solve the appropriate relaxed problem in order to
         evaluate the distance to the goal.
        TODO: implement this method!
        """

        assert isinstance(self.problem, StrictDeliveriesProblem)
        assert isinstance(state, StrictDeliveriesState)

        # raise NotImplemented()  # TODO: remove!

        start_point = state.current_location
        drop_points = self.problem.drop_points - state.dropped_so_far
        if len(drop_points) == 0:
            return 0.0
        gas_stations = self.problem.gas_stations
        tank_capacity = self.problem.gas_tank_capacity
        initial_gas = state.fuel
        relaxed_deliveries_prob_input = DeliveriesProblemInput(
            'inner-problem', start_point, drop_points, gas_stations, tank_capacity, initial_gas)
        relaxed_deliveries_prob = RelaxedDeliveriesProblem(relaxed_deliveries_prob_input)
        inner_problem_solver = AStar(MSTAirDistHeuristic)
        inner_problem_result = inner_problem_solver.solve_problem(relaxed_deliveries_prob)
        if inner_problem_result.final_search_node is None:
            return np.Inf
        return inner_problem_result.final_search_node.cost

