from typing import *
from dataclasses import dataclass
from enum import Enum

from framework import *
from .map_heuristics import AirDistHeuristic
from .cached_map_distance_finder import CachedMapDistanceFinder
from .mda_problem_input import *


__all__ = ['MDAState', 'MDACost', 'MDAProblem', 'MDAOptimizationObjective']


@dataclass(frozen=True)
class MDAState(GraphProblemState):
    """
    An instance of this class represents a state of deliveries problem.
    This state includes the deliveries which are currently loaded on the
     truck, the deliveries which had already been dropped, and the current
     location of the truck (which is either the initial location or the
     last pick/drop location.
    """

    nr_matoshim_on_ambulance: int
    visited_labs: FrozenSet[Laboratory]
    tests_on_ambulance: FrozenSet[ApartmentWithSymptomsReport]
    tests_transferred_to_lab: FrozenSet[ApartmentWithSymptomsReport]
    current_site: Union[Junction, Laboratory, ApartmentWithSymptomsReport]

    @property
    def current_location(self):
        if isinstance(self.current_site, ApartmentWithSymptomsReport) or isinstance(self.current_site, Laboratory):
            return self.current_site.location
        assert isinstance(self.current_site, Junction)
        return self.current_site

    def get_current_location_short_description(self) -> str:
        if isinstance(self.current_site, ApartmentWithSymptomsReport):
            return f'test @ {self.current_site.reporter_name}'
        if isinstance(self.current_site, Laboratory):
            return f'lab {self.current_site.name}'
        return 'initial-location'

    def __str__(self):
        return f'(tests transferred to lab: {list(self.tests_transferred_to_lab)} ' \
               f'tests on ambulance: {list(self.tests_on_ambulance)} ' \
               f'ambulance loc: {self.get_current_location_short_description()})'

    def __eq__(self, other):
        """
        This method is used to determine whether two given state objects represent the same state.
        """
        assert isinstance(other, MDAState)

        # TODO [Ex.xx]: Complete the implementation of this method!
        #  Note that you can simply compare two instances of `Junction` type
        #   (using equals `==` operator) because the class `Junction` explicitly
        #   implements the `__eq__()` method. The types `frozenset`, `ApartmentWithSymptomsReport`, `Laboratory`
        #   are also comparable (in the same manner).
        # raise NotImplementedError  # TODO: remove this line.

        return self.tests_on_ambulance == other.tests_on_ambulance \
               and self.tests_transferred_to_lab == other.tests_transferred_to_lab \
               and self.current_site == other.current_site

    def __hash__(self):
        """
        This method is used to create a hash of a state instance.
        The hash of a state being is used whenever the state is stored as a key in a dictionary
         or as an item in a set.
        It is critical that two objects representing the same state would have the same hash!
        """
        return hash((self.tests_on_ambulance, self.tests_transferred_to_lab, self.current_location))

    def get_total_nr_tests_taken_and_stored_on_ambulance(self) -> int:
        """
        This method returns the total number of packages that are loaded on the truck in this state.
        TODO [Ex.xx]: Implement this method.
         Notice that this method can be implemented using a single line of code - do so!
         Use python's built-it `sum()` function.
         Notice that `sum()` can receive an *ITERATOR* as argument; That is, you can simply write something like this:
        >>> sum(<some expression using item> for item in some_collection_of_items)
        """
        # raise NotImplementedError  # TODO: remove this line.
        return sum(report.nr_roommates for report in self.tests_on_ambulance)


class MDAOptimizationObjective(Enum):
    Distance = 'Distance'
    TestsTravelDistance = 'TestsTravelDistance'


@dataclass(frozen=True)
class MDACost(ExtendedCost):
    """
    An instance of this class is returned as an operator cost by the method
     `MDAProblem.expand_state_with_costs()`.
    The `SearchNode`s that will be created during the run of the search algorithm are going
     to have instances of `MDACost` in SearchNode's `cost` field (instead of float value).
    The reason for using a custom type for the cost (instead of just using a `float` scalar),
     is because we want the cumulative cost (of each search node and particularly of the final
     node of the solution) to be consisted of 2 objectives: (i) distance, and (ii) time
    The field `optimization_objective` controls the objective of the problem (the cost we want
     the solver to minimize). In order to tell the solver which is the objective to optimize,
     we have the `get_g_cost()` method, which returns a single `float` scalar which is only the
     cost to optimize.
    This way, whenever we get a solution, we can inspect the 2 different costs of that solution,
     even though the objective was only one of the costs (time for example).
    Having said that, note that during this assignment we will mostly use the distance objective.
    """
    distance_cost: float = 0.0
    tests_travel_distance_cost: float = 0.0
    optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Distance

    def __add__(self, other):
        assert isinstance(other, MDACost)
        assert other.optimization_objective == self.optimization_objective
        return MDACost(
            optimization_objective=self.optimization_objective,
            distance_cost=self.distance_cost + other.distance_cost,
            tests_travel_distance_cost=self.tests_travel_distance_cost + other.tests_travel_distance_cost)

    def get_g_cost(self) -> float:
        if self.optimization_objective == MDAOptimizationObjective.Distance:
            return self.distance_cost
        assert self.optimization_objective == MDAOptimizationObjective.TestsTravelDistance
        return self.tests_travel_distance_cost

    def __repr__(self):
        return f'MDACost(' \
               f'dist={self.distance_cost:11.3f} meter, ' \
               f'time={self.tests_travel_distance_cost:11.3f} minutes)'


class MDAProblem(GraphProblem):
    """
    An instance of this class represents an MDA problem.
    """

    name = 'MDA-StaffSol'

    def __init__(self,
                 problem_input: MDAProblemInput,
                 streets_map: StreetsMap,
                 optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Distance):
        self.name += f'({problem_input.input_name}({len(problem_input.reported_apartments)}):{optimization_objective.name})'
        initial_state = MDAState(
            nr_matoshim_on_ambulance=problem_input.ambulance.initial_nr_matoshim,
            visited_labs=frozenset(),
            tests_on_ambulance=frozenset(),
            tests_transferred_to_lab=frozenset(),
            current_site=problem_input.ambulance.initial_location)
        super(MDAProblem, self).__init__(initial_state)
        self.problem_input = problem_input
        self.streets_map = streets_map
        self.map_distance_finder = CachedMapDistanceFinder(
            streets_map, AStar(AirDistHeuristic))
        self.optimization_objective = optimization_objective

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[OperatorResult]:
        """
        TODO [Ex.xx]: Implement this method!
        This method represents the `Succ: S -> P(S)` function of the deliveries truck problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The deliveries truck problem operators are defined in the assignment instructions.
        It receives a state and iterates over its successor states.
        Notice that this its return type is an *Iterator*. It means that this function is not
         a regular function, but a `generator function`. Hence, it should be implemented using
         the `yield` statement.
        For each successor, an object of type `OperatorResult` is yielded. This object describes the
            successor state, the cost of the applied operator and its name. Look for its definition
            and use the correct fields in its c'tor. The operator name should be in the following
            format: `pick ClientName` (with the correct client name) if a pick operator was applied,
            or `drop ClientName` if a drop operator was applied. The report object stores its
            reporter name in one of its fields.
        Things you might want to use:
            - The method `self.get_total_nr_packages_loaded()`.
            - The field `self.problem_input.delivery_truck.max_nr_loaded_packages`.
            - The method `self.get_deliveries_waiting_to_pick()` here.
            - The method `self.map_distance_finder.get_map_cost_between()` to calculate
              the operator cost. Its returned value is the operator cost (as is).
            - The c'tor for `AmbulanceState` to create the new successor state.
            - Python's built-in method `frozenset()` to create a new frozen set (for fields that
              expect this type).
            - Other fields of the state and the problem input.
            - Python's sets union operation (`some_set_or_frozenset | some_other_set_or_frozenset`).
        """

        assert isinstance(state_to_expand, MDAState)
        # raise NotImplementedError  # TODO: remove this line!

        # go to a reported apartment
        available_space_in_truck = self.problem_input.ambulance.taken_tests_storage_capacity - state_to_expand.get_total_nr_tests_taken_and_stored_on_ambulance()
        for reported_apartment in self.get_reported_apartments_waiting_to_visit(state_to_expand):
            if available_space_in_truck < reported_apartment.nr_roommates or \
                    state_to_expand.nr_matoshim_on_ambulance < reported_apartment.nr_roommates:
                continue
            new_state = MDAState(
                nr_matoshim_on_ambulance=state_to_expand.nr_matoshim_on_ambulance - reported_apartment.nr_roommates,
                visited_labs=state_to_expand.visited_labs,
                tests_on_ambulance=frozenset(state_to_expand.tests_on_ambulance | {reported_apartment}),
                tests_transferred_to_lab=state_to_expand.tests_transferred_to_lab,
                current_site=reported_apartment)
            yield OperatorResult(
                successor_state=new_state,
                operator_cost=self.get_operator_cost(state_to_expand, new_state),
                operator_name=f'take test from {reported_apartment.reporter_name}')

        # Go to lab
        for laboratory in self.problem_input.laboratories:
            nr_available_matoshim_in_lab = laboratory.max_nr_matoshim * int(laboratory not in state_to_expand.visited_labs)
            if nr_available_matoshim_in_lab < 1 and len(state_to_expand.tests_on_ambulance) < 1:
                continue
            new_state = MDAState(
                nr_matoshim_on_ambulance=state_to_expand.nr_matoshim_on_ambulance + nr_available_matoshim_in_lab,
                visited_labs=frozenset(state_to_expand.visited_labs | {laboratory}),
                tests_on_ambulance=frozenset(),
                tests_transferred_to_lab=frozenset(
                    state_to_expand.tests_transferred_to_lab | state_to_expand.tests_on_ambulance),
                current_site=laboratory)
            yield OperatorResult(
                successor_state=new_state,
                operator_cost=self.get_operator_cost(state_to_expand, new_state),
                operator_name=f'go to lab {laboratory.name}')

    def get_operator_cost(self, prev_state: MDAState, succ_state: MDAState) -> MDACost:
        """
        TODO [staff]: instructions!
        """
        map_distance = self.map_distance_finder.get_map_cost_between(
            prev_state.current_location, succ_state.current_location)
        time_cost = map_distance * len(prev_state.tests_on_ambulance)
        return MDACost(distance_cost=map_distance, tests_travel_distance_cost=time_cost,
                       optimization_objective=self.optimization_objective)

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        TODO [Ex.xx]: implement this method using a single `return` line!
         Use sets/frozensets comparison (`some_set == some_other_set`).
         In order to create a set from some other collection (list/tuple) you can just `set(some_other_collection)`.
        """
        assert isinstance(state, MDAState)
        # raise NotImplementedError  # TODO: remove the line!
        return state.tests_transferred_to_lab == set(self.problem_input.reported_apartments)

    def get_zero_cost(self) -> Cost:
        """
        Overridden method of base class `GraphProblem`. For more information, read
         documentation in the default implementation of this method there.
        In this problem the accumulated cost is not a single float scalar, but an
         extended cost, which actually includes 2 scalar costs.
        """
        return MDACost(optimization_objective=self.optimization_objective)

    def get_reported_apartments_waiting_to_visit(self, state: MDAState) -> Set[ApartmentWithSymptomsReport]:
        """
        This method returns a set of all deliveries that haven't been neither picked nor dropped yet.
        TODO [Ex.xx]: Implement this method.
            Use sets difference operation (`some_set - some_other_set`).
            Note: Given a collection of items, you can create a new set of these items simply by
                `set(my_collection_of_items)`. Then you can use set operations over this newly
                generated set.
            Note: This method can be implemented using a single line of code. Try to do so.
        """
        # raise NotImplementedError  # TODO: remove this line!
        return (set(self.problem_input.reported_apartments) - state.tests_transferred_to_lab) - state.tests_on_ambulance

    def get_all_certain_junctions_in_remaining_ambulance_path(self, state: MDAState) -> List[Junction]:
        """
        This method returns a list of junctions that are part of the remaining route of the ambulance.
        This includes the ambulance's current location, and the locations of the reported apartments
         that hasn't been visited yet.
        The list should be ordered by the junctions index ascendingly (small to big).
        TODO [Ex.xx]: Implement this method.
            Use the method `self.get_reported_apartments_waiting_to_visit(state)`.
            Use python's `sorted(..., key=...)` function.
        """
        # raise NotImplementedError  # TODO: remove this line!
        return sorted(
            [report.location for report in self.get_reported_apartments_waiting_to_visit(state)] + \
            [state.current_location], key=lambda junc: junc.index)
