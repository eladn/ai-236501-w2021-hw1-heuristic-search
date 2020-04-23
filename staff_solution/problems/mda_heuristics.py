import numpy as np
import networkx as nx
from typing import *

from framework import *
from .mda_problem import *
from .cached_air_distance_calculator import CachedAirDistanceCalculator


__all__ = ['MDAMaxAirDistHeuristic', 'MDASumAirDistHeuristic',
           'MDAMSTAirDistHeuristic', 'MDASumAirDistHeuristicForTests',
           'MDATestsTravelDistToNearestLabHeuristic']


class MDATestsTravelDistToNearestLabHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-TimeObjectiveSumOfMinAirDistFromLab-StaffSol'

    def __init__(self, problem: GraphProblem):
        super(MDATestsTravelDistToNearestLabHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.TestsTravelDistance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        TODO: write here instructions & explanations.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        def air_dist_to_closest_lab(junc: Junction) -> float:
            return min(
                self.cached_air_distance_calculator.get_air_distance_between_junctions(junc, lab.location)
                for lab in self.problem.problem_input.laboratories)

        dist_for_cur_state = len(state.tests_on_ambulance) * air_dist_to_closest_lab(state.current_location)
        return dist_for_cur_state + sum(
            air_dist_to_closest_lab(reported_apartment.location)
            for reported_apartment in set(self.problem.problem_input.reported_apartments) -
            (state.tests_on_ambulance | state.tests_transferred_to_lab))


class MDAMaxAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-MaxAirDist-StaffSol'

    def __init__(self, problem: GraphProblem):
        super(MDAMaxAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This method calculated a lower bound of the cost of the remaining path of the truck,
         by calculating the maximum distance within the group of air distances between each
         two junctions in the remaining truck path.

        TODO [Ex.xx]:
            Calculate the `total_distance_lower_bound` by taking the maximum over the group
                {airDistanceBetween(j1,j2) | j1,j2 in JunctionsInRemainingTruckPath s.t. j1 != j2}
            Use the method `get_all_junctions_in_remaining_truck_path()` of the MDA problem.
            Notice: The problem is accessible via the `self.problem` field.
            Use `self.cached_air_distance_calculator.get_air_distance_between_junctions()` for air
                distance calculations.
            Use python's built-in `max()` function. Note that `max()` can receive an *ITERATOR*
                and return the item with the maximum value within this iterator.
            That is, you can simply write something like this:
        >>> max(<some expression using item1 & item2>
        >>>     for item1 in some_items_collection
        >>>     for item2 in some_items_collection
        >>>     if <some condition over item1 & item2>)
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        all_junctions_in_remaining_truck_path = self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)
        if len(all_junctions_in_remaining_truck_path) < 2:
            return 0

        # return 10  # TODO: modify this line.
        return max(
            self.cached_air_distance_calculator.get_air_distance_between_junctions(junction1, junction2)
            for junction1 in all_junctions_in_remaining_truck_path
            for junction2 in all_junctions_in_remaining_truck_path
            if junction1 != junction2)


class MDASumAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-Sum-AirDist-StaffSol'

    def __init__(self, problem: GraphProblem):
        super(MDASumAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic evaluates the distance of the remaining truck route in the following way:
        It builds a path that starts in the current truck's location, and each next junction in
         the path is the (air-distance) nearest junction (to the previous one in the path) among
         all junctions (in `all_junctions_in_remaining_truck_path`) that haven't been visited yet.
        The remaining distance estimation is the cost of this built path.
        Note that we ignore here the problem constraints (like picking before dropping and maximum number of packages
         on the truck). We only make sure to visit all junctions in `all_junctions_in_remaining_truck_path`.
        TODO [Ex.xx]:
            Complete the implementation of this method.
            Use `self.cached_air_distance_calculator.get_air_distance_between_junctions()` for air
             distance calculations.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        all_junctions_in_remaining_truck_path = self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)
        all_junctions_in_remaining_truck_path = set(all_junctions_in_remaining_truck_path)

        if len(all_junctions_in_remaining_truck_path) < 2:
            return 0

        # raise NotImplementedError  # TODO: remove this line and complete the missing part here!

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

        return total_cost_of_greedily_built_path


class MDASumAirDistHeuristicForTests(HeuristicFunction):
    heuristic_name = 'MDA-Sum-AirDist-ForTests-StaffSol'

    def __init__(self, problem: GraphProblem):
        super(MDASumAirDistHeuristicForTests, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic evaluates the distance of the remaining truck route in the following way:
        It builds a path that starts in the current truck's location, and each next junction in
         the path is the (air-distance) nearest junction (to the previous one in the path) among
         all junctions (in `all_junctions_in_remaining_truck_path`) that haven't been visited yet.
        The remaining distance estimation is the cost of this built path.
        Note that we ignore here the problem constraints (like picking before dropping and maximum number of packages
         on the truck). We only make sure to visit all junctions in `all_junctions_in_remaining_truck_path`.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        all_junctions_in_remaining_truck_path = self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)
        all_junctions_in_remaining_truck_path = list(all_junctions_in_remaining_truck_path)
        all_junctions_in_remaining_truck_path.sort(key=lambda junction: junction.index)

        if len(all_junctions_in_remaining_truck_path) < 2:
            return 0

        last_location = state.current_location
        total_cost_of_greedily_built_path = 0
        while len(all_junctions_in_remaining_truck_path) > 1:
            all_junctions_in_remaining_truck_path.remove(last_location)
            next_location = min(
                all_junctions_in_remaining_truck_path,
                key=lambda loc: (self.cached_air_distance_calculator.get_air_distance_between_junctions(last_location, loc), loc.index))
            total_cost_of_greedily_built_path += self.cached_air_distance_calculator.get_air_distance_between_junctions(last_location, next_location)
            last_location = next_location

        return total_cost_of_greedily_built_path


class MDAMSTAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-MST-AirDist-StaffSol'

    def __init__(self, problem: GraphProblem):
        super(MDAMSTAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound for the remaining cost, that is based on a lower bound
         of the distance of the remaining route of the truck. Here this remaining distance is bounded
         (from below) by the weight of the minimum-spanning-tree of the graph in-which the vertices
         are the junctions in the remaining truck route, and the edges weights (edge between each
         junctions pair) are the air-distances between the junctions.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        return self._calculate_junctions_mst_weight_using_air_distance(
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state))

    def _calculate_junctions_mst_weight_using_air_distance(self, junctions: List[Junction]) -> float:
        """
        TODO [Ex.xx]: Implement this method.
              Use `networkx` (nx) package (already imported in this file) to calculate the weight
               of the minimum-spanning-tree of the graph in which the vertices are the given junctions
               and there is an edge between each pair of distinct junctions (no self-loops) for which
               the weight is the air distance between these junctions.
              Use the method `self.cached_air_distance_calculator.get_air_distance_between_junctions()`
               to calculate the air distance between the two junctions.
              Google for how to use `networkx` package for this purpose.
              Use `nx.minimum_spanning_tree()` to get an MST. Calculate the MST size using the method
              `.size(weight='weight')`. Do not manually sum the edges' weights.
        """
        # raise NotImplementedError  # TODO: remove this line!

        junctions_graph = nx.Graph()
        for junction1 in junctions:
            for junction2 in junctions:
                if junction1 == junction2:
                    continue
                junctions_graph.add_edge(
                    junction1, junction2,
                    weight=self.cached_air_distance_calculator.get_air_distance_between_junctions(junction1, junction2))
        junctions_mst = nx.minimum_spanning_tree(junctions_graph)
        return junctions_mst.size(weight='weight')
        # return sum(d['weight'] for (u, v, d) in junctions_mst.edges(data=True))
