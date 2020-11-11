import numpy as np
import networkx as nx
from typing import *

from framework import *
from .mda_problem import *
from .cached_air_distance_calculator import CachedAirDistanceCalculator


__all__ = ['MDAMaxAirDistHeuristic', 'MDASumAirDistHeuristic',
           'MDAMSTAirDistHeuristic', 'MDASumAirDistHeuristicForTests',
           'MDATestsTravelDistToNearestLabHeuristic']


class MDAMaxAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-Max-AirDist-StaffSol'

    def __init__(self, problem: GraphProblem):
        super(MDAMaxAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This method calculated a lower bound of the distance of the remaining path of the ambulance,
         by calculating the maximum distance within the group of air distances between each two
         junctions in the remaining ambulance path. We don't consider laboratories here because we
         do not know what laboratories would be visited in an optimal solution.

        TODO [Ex.21]:
            Calculate the `total_distance_lower_bound` by taking the maximum over the group
                {airDistanceBetween(j1,j2) | j1,j2 in CertainJunctionsInRemainingAmbulancePath s.t. j1 != j2}
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

        all_certain_junctions_in_remaining_ambulance_path = \
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)
        if len(all_certain_junctions_in_remaining_ambulance_path) < 2:
            return 0

        # return 10  # TODO: modify this line.
        return max(
            self.cached_air_distance_calculator.get_air_distance_between_junctions(junction1, junction2)
            for junction1 in all_certain_junctions_in_remaining_ambulance_path
            for junction2 in all_certain_junctions_in_remaining_ambulance_path
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
        This heuristic evaluates the distance of the remaining ambulance route in the following way:
        It builds a path that starts in the current ambulance's location, and each next junction in
         the path is the (air-distance) nearest junction (to the previous one in the path) among
         all certain junctions (in `all_certain_junctions_in_remaining_ambulance_path`) that haven't
         been visited yet.
        The remaining distance estimation is the cost of this built path.
        Note that we ignore here the problem constraints (like enforcing the #matoshim and free
         space in the ambulance's fridge). We only make sure to visit all certain junctions in
         `all_certain_junctions_in_remaining_ambulance_path`.
        TODO [Ex.24]:
            Complete the implementation of this method.
            Use `self.cached_air_distance_calculator.get_air_distance_between_junctions()` for air
             distance calculations.
            For determinism, while building the path, when searching for the next nearest junction,
             use the junction's index as a secondary grading factor. So that if there are 2 different
             junctions with the same distance to the last junction of the so-far-built path, the
             junction to be chosen is the one with the minimal index.
            You might want to use python's tuples comparing to that end.
             Example: (a1, a2) < (b1, b2) iff a1 < b1 or (a1 == b1 and a2 < b2).
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        all_certain_junctions_in_remaining_ambulance_path = \
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)

        if len(all_certain_junctions_in_remaining_ambulance_path) < 2:
            return 0

        # raise NotImplementedError  # TODO: remove this line and complete the missing part here!

        last_location = state.current_location
        total_cost_of_greedily_built_path = 0
        while len(all_certain_junctions_in_remaining_ambulance_path) > 1:
            all_certain_junctions_in_remaining_ambulance_path.remove(last_location)

            # locs_and_dist = [
            #     (loc, (self.cached_air_distance_calculator.get_air_distance_between_junctions(last_location, loc), loc.index))
            #     for loc in all_certain_junctions_in_remaining_ambulance_path]
            # min_dist_idx = np.argmin(np.array([dist for _, dist in locs_and_dist]))
            # next_location = locs_and_dist[min_dist_idx][0]

            next_location = min(
                all_certain_junctions_in_remaining_ambulance_path,
                key=lambda loc: (
                self.cached_air_distance_calculator.get_air_distance_between_junctions(last_location, loc), loc.index))

            total_cost_of_greedily_built_path += self.cached_air_distance_calculator.get_air_distance_between_junctions(
                last_location, next_location)
            last_location = next_location

        return total_cost_of_greedily_built_path


class MDASumAirDistHeuristicForTests(HeuristicFunction):
    """
    Used for tests only (for determinism).
    """
    heuristic_name = 'MDA-Sum-AirDist-ForTests-StaffSol'

    def __init__(self, problem: GraphProblem):
        super(MDASumAirDistHeuristicForTests, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        Used for tests only (for determinism).
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        all_certain_junctions_in_remaining_ambulance_path = \
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)
        all_certain_junctions_in_remaining_ambulance_path = list(all_certain_junctions_in_remaining_ambulance_path)
        all_certain_junctions_in_remaining_ambulance_path.sort(key=lambda junction: junction.index)

        if len(all_certain_junctions_in_remaining_ambulance_path) < 2:
            return 0

        last_location = state.current_location
        total_cost_of_greedily_built_path = 0
        while len(all_certain_junctions_in_remaining_ambulance_path) > 1:
            all_certain_junctions_in_remaining_ambulance_path.remove(last_location)
            next_location = min(
                all_certain_junctions_in_remaining_ambulance_path,
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
        This heuristic returns a lower bound for the distance of the remaining route of the ambulance.
        Here this remaining distance is bounded (from below) by the weight of the minimum-spanning-tree
         of the graph, in-which the vertices are the junctions in the remaining ambulance route, and the
         edges weights (edge between each junctions pair) are the air-distances between the junctions.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        return self._calculate_junctions_mst_weight_using_air_distance(
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state))

    def _calculate_junctions_mst_weight_using_air_distance(self, junctions: List[Junction]) -> float:
        """
        TODO [Ex.27]: Implement this method.
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


class MDATestsTravelDistToNearestLabHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-TimeObjectiveSumOfMinAirDistFromLab-StaffSol'

    def __init__(self, problem: GraphProblem):
        super(MDATestsTravelDistToNearestLabHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.TestsTravelDistance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound to the remained tests-travel-distance of the remained ambulance path.
        The main observation is that driving from a laboratory to a reported-apartment does not increase the
         tests-travel-distance cost. So the best case (lowest cost) is when we go to the closest laboratory right
         after visiting any reported-apartment.
        If the ambulance currently stores tests, this total remained cost includes the #tests_on_ambulance times
         the distance from the current ambulance location to the closest lab.
        The rest part of the total remained cost includes the distance between each non-visited reported-apartment
         and the closest lab (to this apartment) times the roommates in this apartment (as we take tests for all
         roommates).
        TODO [Ex.33]:
            Complete the implementation of this method.
            Use `self.problem.get_reported_apartments_waiting_to_visit(state)`.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        def air_dist_to_closest_lab(junction: Junction) -> float:
            """
            Returns the distance between `junction` and the laboratory that is closest to `junction`.
            """
            # return min(...)  # TODO: replace `...` with the relevant implementation.
            return min(
                self.cached_air_distance_calculator.get_air_distance_between_junctions(junction, lab.location)
                for lab in self.problem.problem_input.laboratories)

        # raise NotImplementedError
        dist_for_cur_state = state.get_total_nr_tests_taken_and_stored_on_ambulance() * air_dist_to_closest_lab(state.current_location)
        return dist_for_cur_state + sum(
            air_dist_to_closest_lab(reported_apartment.location) * reported_apartment.nr_roommates
            for reported_apartment in self.problem.get_reported_apartments_waiting_to_visit(state))
