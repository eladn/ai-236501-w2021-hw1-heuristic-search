import numpy as np
import networkx as nx
from typing import *

from framework import *
from .deliveries_truck_problem import *
from .cached_air_distance_calculator import CachedAirDistanceCalculator


__all__ = ['TruckDeliveriesMaxAirDistHeuristic', 'TruckDeliveriesSumAirDistHeuristic',
           'TruckDeliveriesMSTAirDistHeuristic']


class TruckDeliveriesMaxAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'TruckDeliveriesMaxAirDist'

    def __init__(self, problem: GraphProblem):
        super(TruckDeliveriesMaxAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesTruckProblem)
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This method calculated a lower bound of the cost of the remaining path of the truck,
         by calculating the maximum distance within the group of air distances between each
         two junctions in the remaining truck path.

        TODO:
            Calculate the `total_distance_lower_bound` by taking the maximum over the group
                {airDistanceBetween(j1,j2) | j1,j2 in JunctionsInRemainingTruckPath s.t. j1 != j2}
            Use python's built-in `max()` function. Note that `max()` can receive an *ITERATOR*
                and return the item with the maximum value within this iterator.
            That is, you can simply write something like this:
        >>> max(<some expression using item1 & item2>
        >>>     for item1 in some_items_collection
        >>>     for item2 in some_items_collection
        >>>     if <some condition over item1 & item2>)
            Use `self.cached_air_distance_calculator.get_air_distance_between_junctions()` for air
                distance calculations.
        """
        assert isinstance(self.problem, DeliveriesTruckProblem)
        assert isinstance(state, DeliveriesTruckState)

        all_junctions_in_remaining_truck_path = self.problem.get_all_junctions_in_remaining_truck_path(state)
        if len(all_junctions_in_remaining_truck_path) < 2:
            return 0

        total_distance_lower_bound = 10  # TODO: modify this line.
        total_distance_lower_bound = max(
            self.cached_air_distance_calculator.get_air_distance_between_junctions(junction1, junction2)
            for junction1 in all_junctions_in_remaining_truck_path
            for junction2 in all_junctions_in_remaining_truck_path
            if junction1 != junction2)

        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)


class TruckDeliveriesSumAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'TruckDeliveriesSumAirDist'

    def __init__(self, problem: GraphProblem):
        super(TruckDeliveriesSumAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesTruckProblem)
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic evaluates the distance of the remaining truck route in the following way:
        It builds a path that starts in the current truck's location, and each next junction in
         the path is the (air-distance) nearest junction (to the previous one in the path) among
         all junctions (in `all_junctions_in_remaining_truck_path`) that haven't been visited yet.
        The estimation
        TODO:
            Complete the implementation of this method.
            Use `self.cached_air_distance_calculator.get_air_distance_between_junctions()` for air
             distance calculations.
        """
        assert isinstance(self.problem, DeliveriesTruckProblem)
        assert isinstance(state, DeliveriesTruckState)

        all_junctions_in_remaining_truck_path = self.problem.get_all_junctions_in_remaining_truck_path(state)

        if len(all_junctions_in_remaining_truck_path) < 2:
            return 0

        # raise NotImplemented()  # TODO: remove this line and complete the missing part here!

        last_location = state.current_location
        total_cost_of_greedily_built_path = 0
        while len(all_junctions_in_remaining_truck_path) > 1:
            all_junctions_in_remaining_truck_path.remove(last_location)
            locs_and_dist = [
                (loc, self.cached_air_distance_calculator.get_air_distance_between_junctions(last_location, loc))
                for loc in all_junctions_in_remaining_truck_path]
            min_dist_idx = np.argmin(np.array([dist for _, dist in locs_and_dist]))
            next_location = locs_and_dist[min_dist_idx][0]
            total_cost_of_greedily_built_path += self.cached_air_distance_calculator.get_air_distance_between_junctions(last_location, next_location)
            last_location = next_location

        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_cost_of_greedily_built_path)


class TruckDeliveriesMSTAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'TruckDeliveriesMSTAirDist'

    def __init__(self, problem: GraphProblem):
        super(TruckDeliveriesMSTAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesTruckProblem)
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound for the remaining cost, that is based on a lower bound
         of the distance of the remaining route of the truck. Here this remaining distance is bounded
         (from below) by the weight of the minimum-spanning-tree of the graph in-which the vertices
         are the junctions in the remaining truck route, and the edges weights (edge between each
         junctions pair) are the air-distances between the junctions.
        """
        assert isinstance(self.problem, DeliveriesTruckProblem)
        assert isinstance(state, DeliveriesTruckState)

        total_distance_lower_bound = self._calculate_junctions_mst_weight_using_air_distance(
            self.problem.get_all_junctions_in_remaining_truck_path(state))
        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)

    def _calculate_junctions_mst_weight_using_air_distance(self, junctions: Set[Junction]) -> float:
        """
        TODO: Implement this method.
              Use `networkx` (nx) package (already imported in this file) to calculate the weight
               of the minimum-spanning-tree of the graph in which the vertices are the given junctions
               and there is an edge between each pair of distinct junctions (no self-loops) for which
               the weight is the air distance between these junctions.
              Use the method `self.cached_air_distance_calculator.get_air_distance_between_junctions()`
               to calculate the air distance between the two junctions.
              Google for how to use `networkx` package for this purpose.
        """
        # raise NotImplemented()  # TODO: remove this line!

        junctions_graph = nx.Graph()
        idx_to_junction = {idx: vertex for idx, vertex in enumerate(junctions)}
        for junction1_idx, junction1 in idx_to_junction.items():
            for junction2_idx, junction2 in idx_to_junction.items():
                if junction1_idx == junction2_idx:
                    continue
                junctions_graph.add_edge(
                    junction1_idx, junction2_idx,
                    weight=self.cached_air_distance_calculator.get_air_distance_between_junctions(junction1, junction2))
        junctions_mst = nx.minimum_spanning_tree(junctions_graph)
        return sum(d['weight'] for (u, v, d) in junctions_mst.edges(data=True))
