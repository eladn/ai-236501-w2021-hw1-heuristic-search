from typing import *
from dataclasses import dataclass
import numpy as np
import networkx as nx

from framework import *
from .map_problem import MapProblem, MapState
from .cached_map_distance_finder import CachedMapDistanceFinder
from .deliveries_problem_input import *
from .cached_air_distance_calculator import CachedAirDistanceCalculator


@dataclass(frozen=True)
class DeliveriesTruckState(GraphProblemState):
    """
    An instance of this class represents a state of deliveries problem.
    This state includes the deliveries which are currently loaded on the
     truck, the deliveries which had already been dropped, and the current
     location of the truck (which is either the initial location or the
     last pick/drop location.
    """

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
        """
        This method is used to determine whether two given state objects represent the same state.
        """
        assert isinstance(other, DeliveriesTruckState)

        # TODO: Complete the implementation of this method!
        #       Note that you can simply compare two instances of `Junction` type
        #        (using equals `==` operator) because the class `Junction` explicitly
        #        implements the `__eq__()` method. The types `frozenset` and `Delivery`
        #        are also comparable (in the same manner).
        # raise NotImplemented()  # TODO: remove this line.

        return self.loaded_deliveries == other.loaded_deliveries \
               and self.dropped_deliveries == other.dropped_deliveries \
               and self.current_location == other.current_location

    def __hash__(self):
        """
        This method is used to create a hash of a state instance.
        The hash of a state being is used whenever the state is stored as a key in a dictionary
         or as an item in a set.
        It is critical that two objects representing the same state would have the same hash!
        """
        return hash((self.loaded_deliveries, self.dropped_deliveries, self.current_location))


@dataclass(frozen=True)
class DeliveryCost(ExtendedCost):
    """
    An instance of this class is returned as an operator cost by the method
     `DeliveriesTruckProblem.expand_state_with_costs()`.
    The reason for using a custom type for the cost (instead of just using a `float` scalar),
     is because we want the cumulative cost (of each search node and particularly of the final
     node of the solution) to be consisted of 3 objectives: (i) distance, (ii) time, and
     (iii) money.
    The field `optimization_objective` controls the objective of the problem (the cost we want
     the solver to minimize). In order to tell the solver which is the objective to optimize,
     we have the `get_g_cost()` method, which returns a single `float` scalar which is only the
     cost to optimize.
    This way, whenever we get a solution, we can inspect the 3 different costs of that solution,
     even though the objective was only one of the costs (time for example).
    Having said that, note that during this assignment we will mostly use the distance objective.
    """
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
    """
    An instance of this class represents a deliveries truck problem.
    """

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
        """
        TODO: implement this method!
        This method represents the `Succ: S -> P(S)` function of the deliveries truck problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The deliveries truck problem operators are defined in the assignment instructions.
        It receives a state and iterates over its successor states.
        Notice that this its return type is an *Iterator*. It means that this function is not
         a regular function, but a `generator function`. Hence, it should be implemented using
         the `yield` statement.
        For each successor state, a pair of the successor state and the operator cost is yielded.
        """

        assert isinstance(state_to_expand, DeliveriesTruckState)
        # raise NotImplemented()  # TODO: remove this line!

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
        """
        TODO: Modify the implementation of this method, so that for a given link (road), it would return
               the extended cost of this link. That is, the distance should remain as it is now, but both
               the `time_cost` and the `money_cost` should be set appropriately.
              Use the `optimal_velocity` and the `gas_cost_per_meter` returned by the method
               `self.problem_input.delivery_truck.calc_optimal_driving_parameters()`, in order to calculate
               the `time_cost` and the `money_cost`.
              Note that the `money_cost` is the total gas cost for this given link plus the total fee paid
               for driving on this road if this road is a toll road. Use the appropriate Link's field to
               check whether is it a tool road and to get the distance of this road, and use the appropriate
               field in the problem input (accessible by `self.problem_input`) to get the toll road cost per
               meter.
        """
        optimal_velocity, gas_cost_per_meter = self.problem_input.delivery_truck.calc_optimal_driving_parameters(
            optimization_objective=self.optimization_objective, max_driving_speed=link.max_speed)
        # return DeliveryCost(
        #     distance_cost=link.distance,
        #     time_cost=0,
        #     money_cost=0,
        #     optimization_objective=self.optimization_objective)  # TODO: modify this!

        return DeliveryCost(
            distance_cost=link.distance,
            time_cost=link.distance / optimal_velocity,
            money_cost=gas_cost_per_meter * link.distance + (self.problem_input.toll_road_cost_per_meter * link.distance if link.is_toll_road else 0),
            optimization_objective=self.optimization_objective)

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        TODO: implement this method!
        """
        assert isinstance(state, DeliveriesTruckState)
        # raise NotImplemented()  # TODO: remove!
        return state.dropped_deliveries == set(self.problem_input.deliveries)

    def get_zero_cost(self) -> Cost:
        return DeliveryCost(optimization_objective=self.optimization_objective)

    def get_cost_lower_bound_from_distance_lower_bound(self, total_distance_lower_bound: float) -> float:
        """
        Used by the heuristics of the deliveries truck problem.
        Given a lower bound of the distance (in meters) that the truck has left to travel,
         this method returns an appropriate lower bound of the distance/time/money cost
         based on the problem's objective.
        TODO: We left only partial implementation of this method (just the trivial distance objective).
              Complete the implementation of this method!
              You might want to use constants like `MIN_ROAD_SPEED` or `MAX_ROAD_SPEED`.
              For the money cost, you would like to use the method `self._calc_map_road_cost()`. This
               method expects to get a `Link` instance and returns the (extended) cost of this road.
               Although the `total_distance_lower_bound` actually represents an estimation for the
               remaining route (and not an actual road on the map), you can simply create a `Link`
               instance (that represents this whole remaining path) for this purpose.
              Remember: The return value should be a real lower bound. This is required for the
               heuristic to be acceptable.
        """
        if self.optimization_objective == OptimizationObjective.Distance:
            return total_distance_lower_bound
        elif self.optimization_objective == OptimizationObjective.Time:
            # raise NotImplemented()  # TODO: remove this line and complete the implementation of this case!
            return total_distance_lower_bound / MAX_ROAD_SPEED
        else:
            assert self.optimization_objective == OptimizationObjective.Money
            # raise NotImplemented()  # TODO: remove this line and complete the implementation of this case!
            lower_bound_for_gas_cost_of_driving_remaining_roads = self._calc_map_road_cost(
                Link(0, 0, total_distance_lower_bound, 0, MAX_ROAD_SPEED, False)).money_cost
            return lower_bound_for_gas_cost_of_driving_remaining_roads


class TruckDeliveriesMaxAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'TruckDeliveriesMaxAirDist'

    def __init__(self, problem: GraphProblem):
        super(TruckDeliveriesMaxAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesTruckProblem)
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, DeliveriesTruckProblem)
        assert isinstance(state, DeliveriesTruckState)

        # raise NotImplemented()  # TODO: remove this line!

        deliveries_waiting_to_pick = (set(
            self.problem.problem_input.deliveries) - state.dropped_deliveries) - state.loaded_deliveries
        all_junctions_to_visit = {delivery.pick_location for delivery in deliveries_waiting_to_pick} | \
                                 {delivery.drop_location for delivery in deliveries_waiting_to_pick} | \
                                 {delivery.drop_location for delivery in state.loaded_deliveries} | \
                                 {state.current_location}
        if len(all_junctions_to_visit) < 2:
            return 0
        total_distance_lower_bound = max(
            self.cached_air_distance_calculator.get_air_distance_between_junctions(junction1, junction2)
            for junction1 in all_junctions_to_visit
            for junction2 in all_junctions_to_visit
            if junction1 != junction2)
        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)


class TruckDeliveriesSumAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'TruckDeliveriesSumAirDist'

    def __init__(self, problem: GraphProblem):
        super(TruckDeliveriesSumAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesTruckProblem)
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, DeliveriesTruckProblem)
        assert isinstance(state, DeliveriesTruckState)

        # raise NotImplemented()  # TODO: remove this line!

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
            locs_and_dist = [(loc, self.cached_air_distance_calculator.get_air_distance_between_junctions(last_location, loc)) for loc in all_junctions_to_visit]
            min_dist_idx = np.argmin(np.array([dist for _, dist in locs_and_dist]))
            next_location = locs_and_dist[min_dist_idx][0]
            total_distance_sum += self.cached_air_distance_calculator.get_air_distance_between_junctions(last_location, next_location)
            last_location = next_location

        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_sum)


class TruckDeliveriesMSTAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'TruckDeliveriesMSTAirDist'

    def __init__(self, problem: GraphProblem):
        super(TruckDeliveriesMSTAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesTruckProblem)
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        TODO: Implement this method.

        """
        assert isinstance(self.problem, DeliveriesTruckProblem)
        assert isinstance(state, DeliveriesTruckState)

        # raise NotImplemented()  # TODO: remove this line!

        deliveries_waiting_to_pick = (set(
            self.problem.problem_input.deliveries) - state.dropped_deliveries) - state.loaded_deliveries
        all_junctions_to_visit = {delivery.pick_location for delivery in deliveries_waiting_to_pick} | \
                                 {delivery.drop_location for delivery in deliveries_waiting_to_pick} | \
                                 {delivery.drop_location for delivery in state.loaded_deliveries} | \
                                 {state.current_location}
        total_distance_lower_bound = self._calculate_junctions_mst_weight_using_air_distance(all_junctions_to_visit)
        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)

    def _calculate_junctions_mst_weight_using_air_distance(self, junctions: Set[Junction]) -> float:
        """
        TODO: Implement this method.
              Use `networkx` (nx) package (already imported in this file) to calculate the weight
               of the minimum-spanning-tree of the graph in which the vertices are the given junctions
               and there is an edge between each pair of dintinct junctions (no self-loops) for which
               the weight is the air distance between these junctions.
              Use the method `self.cached_air_distance_calculator.get_air_distance_between_junctions()`
               to calculate the air distance between the two junctions.
              Google for how to use `networkx` for this purpose.
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
        # raise NotImplemented()  # TODO: remove this line!

        source_junction = self.problem.streets_map[state.junction_id]
        target_junction = self.problem.streets_map[self.problem.target_junction_id]
        total_distance_lower_bound = source_junction.calc_air_distance_from(target_junction)
        return self.outer_deliveries_problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)
