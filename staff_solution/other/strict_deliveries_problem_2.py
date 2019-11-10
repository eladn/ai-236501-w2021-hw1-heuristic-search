from framework import *
from deliveries.map_problem import MapProblem
from deliveries.deliveries_problem_input import DeliveriesProblemInput
from deliveries.relaxed_deliveries_problem import RelaxedDeliveriesState, RelaxedDeliveriesProblem

from typing import Set, FrozenSet, Optional, Iterator, Tuple, Union


TENTATIVE = 'TENTATIVE'
REALIZED = 'REALIZED'


class StrictDeliveriesState_(RelaxedDeliveriesState):
    """
    For advanced clauses (in the question) we would like each state
       to also contain a binary field TENTATIVE/REALIZED.
    When expanding a "realized" state, only tentative states are created.
       These tentative states have fuel = last_state_fuel - AirDist(last_junc, next_junc)
       [If this expression is negative, this next state is not yielded]
       [If the next state is a gas station its fuel = tank_capacity].
       These tentative states have cost = AirDist(last_junc, next_junc).
    When a tentative state is expanded, only one state is yielded - the realized one.
       It is yielded only if it is really possible to reach it from the last realized state.
       Otherwise, no continuation to the tentative state.
       The fuel and cost are the positive deltas from the tentative ones.
    """
    def __init__(self,
                 current_location: Junction,
                 dropped_so_far: Union[Set[Junction], FrozenSet[Junction]],
                 fuel: float,
                 tentative_or_realized: str,
                 last_realized_point: Optional[Junction] = None):
        super(StrictDeliveriesState_, self).__init__(current_location, dropped_so_far, fuel)
        assert (tentative_or_realized in {REALIZED, TENTATIVE})
        self.tentative_or_realized = tentative_or_realized
        self.last_realized_point = last_realized_point

    def __eq__(self, other):
        return super(StrictDeliveriesState_, self).__eq__(other) and \
               self.tentative_or_realized == other.tentative_or_realized

    def __hash__(self):
        return hash((super(StrictDeliveriesState_, self).__hash__(), self.tentative_or_realized))

    def __str__(self):
        return str(self.current_location.index) + ('T' if self.is_tentative else 'R')

    @property
    def is_realized(self) -> bool:
        return self.tentative_or_realized == REALIZED

    @property
    def is_tentative(self) -> bool:
        return self.tentative_or_realized == TENTATIVE


class StrictDeliveriesProblem_(RelaxedDeliveriesProblem):
    """
    An instance of this class represents a strict deliveries problem.
    """

    name = 'StrictDeliveries'

    def __init__(self, problem_input: DeliveriesProblemInput, roads: Roads,
                 inner_problem_solver: GraphProblemSolver, use_cache: bool = True):
        super(StrictDeliveriesProblem_, self).__init__(problem_input)
        self.initial_state = StrictDeliveriesState_(
            problem_input.start_point, set(), problem_input.gas_tank_init_fuel, REALIZED)
        self.roads = roads
        self.inner_problem_solver = inner_problem_solver
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

    def expand_state_with_costs(self, state: GraphProblemState) -> Iterator[Tuple[GraphProblemState, float]]:
        """
        TODO: implement this method!
        This method represents the `Succ: S -> P(S)` function of the strict deliveries problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The relaxed problem operators are defined in the assignment instructions.
        It receives a state and iterates over the successor states.
        Notice that this is an *Iterator*. Hence it should be implemented using the `yield` keyword.
        For each successor, a pair of the successor state and the operator cost is yielded.
        """
        assert isinstance(state, StrictDeliveriesState_)

        # raise NotImplemented()  # TODO: remove!

        if state.is_realized:
            for next_stop_point in self.possible_stop_points:
                if next_stop_point == state.current_location:
                    continue
                if next_stop_point in state.dropped_so_far and next_stop_point not in self.gas_stations:
                    continue
                air_dist = state.current_location.calc_air_distance_from(next_stop_point)
                if air_dist >= state.fuel:
                    continue
                fuel = state.fuel - air_dist
                assert fuel > 0
                if next_stop_point in self.gas_stations:
                    fuel = self.gas_tank_capacity
                dropped_so_far = set(state.dropped_so_far)
                if next_stop_point in self.drop_points:
                    dropped_so_far.add(next_stop_point)
                next_state = StrictDeliveriesState_(
                    next_stop_point, dropped_so_far, fuel, TENTATIVE, state.current_location)
                yield next_state, air_dist
        else:
            assert state.is_tentative
            assert state.last_realized_point is not None
            # Solve map problem from last realized point to here.
            if self._get_from_cache((state.last_realized_point.index, state.current_location.index)) is None:
                inner_map_problem = MapProblem(
                    self.roads, state.last_realized_point.index, state.current_location.index)
                inner_problem_result = self.inner_problem_solver.solve_problem(inner_map_problem)
                if inner_problem_result.final_search_node is None:
                    return  # cannot reach realized point; TODO: is it possible? should we assert it is reachable? This is probably impossible....
                self._insert_to_cache((state.last_realized_point.index, state.current_location.index),
                                      inner_problem_result.final_search_node.cost)
                real_distance_from_last_realized_point = inner_problem_result.final_search_node.cost
            else:
                real_distance_from_last_realized_point = self._get_from_cache(
                    (state.last_realized_point.index, state.current_location.index))
            air_dist_from_last_realized_point = state.current_location.calc_air_distance_from(state.last_realized_point)
            cost_delta = real_distance_from_last_realized_point - air_dist_from_last_realized_point
            assert cost_delta >= 0
            fuel = state.fuel - cost_delta
            if fuel <= 0:
                return  # cannot reach realized point
            next_state = StrictDeliveriesState_(
                state.current_location, state.dropped_so_far, fuel, REALIZED)
            yield next_state, cost_delta

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        TODO: implement this method!
        """
        assert isinstance(state, StrictDeliveriesState_)

        # raise NotImplemented()  # TODO: remove!

        return state.is_realized and state.dropped_so_far == self.drop_points


# StrictDeliveriesState = StrictDeliveriesState_
# StrictDeliveriesProblem = StrictDeliveriesProblem_
