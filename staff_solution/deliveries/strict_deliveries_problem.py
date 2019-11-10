from framework import *
from .map_problem import MapProblem
from .deliveries_problem_input import DeliveriesProblemInput
from .relaxed_deliveries_problem import RelaxedDeliveriesState, RelaxedDeliveriesProblem

from typing import Iterator, Tuple


class StrictDeliveriesState(RelaxedDeliveriesState):
    """
    An instance of this class represents a state of the strict
     deliveries problem.
    This state is basically similar to the state of the relaxed
     problem. Hence, this class inherits from `RelaxedDeliveriesState`.

    TODO:
        If you believe you need to modify the state for the strict
         problem in some sense, please go ahead and do so.
    """
    pass


class StrictDeliveriesProblem(RelaxedDeliveriesProblem):
    """
    An instance of this class represents a strict deliveries problem.
    """

    name = 'StrictDeliveries'

    def __init__(self, problem_input: DeliveriesProblemInput, roads: Roads,
                 inner_problem_solver: GraphProblemSolver, use_cache: bool = True):
        super(StrictDeliveriesProblem, self).__init__(problem_input)
        self.initial_state = StrictDeliveriesState(
            problem_input.start_point, frozenset(), problem_input.gas_tank_init_fuel)
        self.inner_problem_solver = inner_problem_solver
        self.roads = roads
        self.use_cache = use_cache
        self._init_cache()

    def _init_cache(self):
        self._cache = {}
        self.nr_cache_hits = 0
        self.nr_cache_misses = 0

    def _insert_to_cache(self, key, val):
        if self.use_cache:
            self._cache[key] = val

    def _get_from_cache(self, key):
        if not self.use_cache:
            return None
        if key in self._cache:
            self.nr_cache_hits += 1
        else:
            self.nr_cache_misses += 1
        return self._cache.get(key)

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[Tuple[GraphProblemState, float]]:
        """
        TODO: implement this method!
        This method represents the `Succ: S -> P(S)` function of the strict deliveries problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The relaxed problem operators are defined in the assignment instructions.
        It receives a state and iterates over the successor states.
        Notice that this is an *Iterator*. Hence it should be implemented using the `yield` keyword.
        For each successor, a pair of the successor state and the operator cost is yielded.
        """
        assert isinstance(state_to_expand, StrictDeliveriesState)

        # raise NotImplemented()  # TODO: remove!

        for next_stop_point in self.possible_stop_points:
            if next_stop_point == state_to_expand.current_location:
                continue
            if next_stop_point in state_to_expand.dropped_so_far and next_stop_point not in self.gas_stations:
                continue
            air_dist = state_to_expand.current_location.calc_air_distance_from(next_stop_point)
            if air_dist >= state_to_expand.fuel:
                continue

            if self._get_from_cache((state_to_expand.current_location.index, next_stop_point.index)) is None:
                inner_map_problem = MapProblem(
                    self.roads, state_to_expand.current_location.index, next_stop_point.index)
                inner_problem_result = self.inner_problem_solver.solve_problem(inner_map_problem)
                if inner_problem_result.final_search_node is None:
                    continue  # cannot reach realized point
                self._insert_to_cache((state_to_expand.current_location.index, next_stop_point.index),
                                      inner_problem_result.final_search_node.cost)
                real_distance = inner_problem_result.final_search_node.cost
            else:
                real_distance = self._get_from_cache(
                    (state_to_expand.current_location.index, next_stop_point.index))

            if real_distance >= state_to_expand.fuel:
                continue

            fuel = state_to_expand.fuel - real_distance
            assert fuel > 0
            if next_stop_point in self.gas_stations:
                fuel = self.gas_tank_capacity
            dropped_so_far = set(state_to_expand.dropped_so_far)
            if next_stop_point in self.drop_points:
                dropped_so_far.add(next_stop_point)
            next_state = StrictDeliveriesState(
                next_stop_point, dropped_so_far, fuel)
            yield next_state, real_distance

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        TODO: implement this method!
        """
        assert isinstance(state, StrictDeliveriesState)

        # raise NotImplemented()  # TODO: remove!
        return state.dropped_so_far == self.drop_points
