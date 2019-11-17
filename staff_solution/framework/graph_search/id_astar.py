from .graph_problem_interface import *
from .astar import AStar
from .utils.utils import calc_relative_error
from .utils.timer import Timer
from typing import Optional, Tuple
from dataclasses import dataclass
import itertools
import math
import numpy as np


@dataclass(frozen=True)
class DFSFResult:
    """
    An instance of this class is returned for each call of the inner method `_dfs_f()`.
    The DFS-L has many values to return. We use a class to make the return value friendly.
    """

    max_depth_reached: int
    f_limit_for_next_iteration: Optional[float] = None
    nr_expanded_states: int = 0
    solution_cost: Optional[Cost] = None
    solution_path: Optional[Tuple[StatesPathNode, ...]] = None
    stopped_because_max_nr_states_to_expand_reached: bool = False

    @property
    def is_solution_found(self):
        return self.solution_path is not None


class IDAStar(GraphProblemSolver):
    solver_name: str = 'ID-A*'

    def __init__(self,
                 heuristic_function_type: HeuristicFunctionType,
                 heuristic_weight: float = 0.5,
                 deepening_technique: str = 'binary_search',
                 max_cost_relative_error: float = 0.005,
                 max_nr_iterations: Optional[int] = None,
                 max_nr_states_to_expand: Optional[int] = None):
        if deepening_technique not in {'iterative', 'binary_search'}:
            raise ValueError(f"Parameter `deepening_technique` must be on of 'iterative' or 'binary_search'. "
                             f"Got {deepening_technique}.")
        self.heuristic_function_type = heuristic_function_type
        self.heuristic_function = None
        self.heuristic_weight = heuristic_weight
        self.deepening_technique = deepening_technique
        self.max_cost_relative_error = max_cost_relative_error
        self.max_nr_iterations = max_nr_iterations
        self.max_nr_states_to_expand = max_nr_states_to_expand

    def solve_problem(self, problem: GraphProblem) -> SearchResult:
        self.heuristic_function = self.heuristic_function_type(problem)
        self.solver_name = f'{self.__class__.solver_name} (h={self.heuristic_function.heuristic_name}, w={self.heuristic_weight:.3f})'
        with Timer(print_title=False) as timer:
            if self.deepening_technique == 'binary_search':
                search_result = self._binary_search_f_deepening(problem)
            elif self.deepening_technique == 'iterative':
                search_result = self._iterative_f_deepening(problem)
            else:
                raise ValueError(f"Parameter `deepening_technique` must be on of 'iterative' or 'binary_search'. "
                                 f"Got {self.deepening_technique}.")
        return search_result._replace(solving_time=timer.elapsed)

    def _binary_search_f_deepening(self, problem: GraphProblem) -> SearchResult:
        nr_iterations = 0
        total_nr_expanded_states = 0
        max_depth_reached = 0

        # Ensure that there is no solution for f_limit=0 (in order to to use it as `low_f_limit`)
        zero_f_limit_result = self._dfs_f(problem, problem.initial_state, f_limit=0)
        nr_iterations += 1
        total_nr_expanded_states += zero_f_limit_result.nr_expanded_states
        max_depth_reached = max(max_depth_reached, zero_f_limit_result.max_depth_reached)
        if zero_f_limit_result.solution_path is not None:
            return SearchResult(
                solver=self,
                problem=problem,
                solution_path=GraphProblemStatesPath(zero_f_limit_result.solution_path),
                nr_expanded_states=total_nr_expanded_states,
                # nr_expanded_states_for_optimal_f_limit=result.nr_expanded_states,
                max_nr_stored_states=max_depth_reached,
                nr_iterations=nr_iterations)

        # Quickly find some some f_limit for which a solution exists (in order to to use it as `high_f_limit`)
        greedy = AStar(self.heuristic_function_type, heuristic_weight=1)
        greedy_solution = greedy.solve_problem(problem)
        assert greedy_solution.is_solution_found
        greedy_solution_cost = greedy_solution.solution_g_cost
        assert greedy_solution_cost > 0

        # Binary deepening
        low_f_limit = 0  # Not found solution at this f_limit
        high_f_limit = greedy_solution_cost * (1 - self.heuristic_weight)  # Found solution at this f_limit

        max_f_limit_relative_error = self.max_cost_relative_error * (1 - self.heuristic_weight)
        while calc_relative_error(low_f_limit, high_f_limit) > max_f_limit_relative_error:
            nr_iterations += 1
            if self.max_nr_iterations is not None and nr_iterations > self.max_nr_iterations:
                return SearchResult(
                    solver=self,
                    problem=problem,
                    nr_expanded_states=total_nr_expanded_states,
                    max_nr_stored_states=max_depth_reached,
                    nr_iterations=nr_iterations-1,
                    stop_reason=StopReason.ExceededMaxNrIteration)

            assert high_f_limit > low_f_limit
            mid_f_limit = (high_f_limit + low_f_limit) / 2
            # print(f'low_f_limit: {low_f_limit} -- mid_f_limit: {mid_f_limit} -- high_f_limit: {high_f_limit}')
            max_nr_states_to_expand = None if self.max_nr_states_to_expand is None \
                else self.max_nr_states_to_expand - total_nr_expanded_states
            result = self._dfs_f(problem=problem, state=problem.initial_state, f_limit=mid_f_limit,
                                 max_nr_states_to_expand=max_nr_states_to_expand)
            total_nr_expanded_states += result.nr_expanded_states
            max_depth_reached = max(max_depth_reached, result.max_depth_reached)

            if result.stopped_because_max_nr_states_to_expand_reached:
                return SearchResult(
                    solver=self,
                    problem=problem,
                    nr_expanded_states=total_nr_expanded_states,
                    max_nr_stored_states=max_depth_reached,
                    nr_iterations=nr_iterations,
                    stop_reason=StopReason.ExceededMaxNrStatesToExpand)

            if result.is_solution_found:
                high_f_limit = mid_f_limit
            else:
                low_f_limit = mid_f_limit

        assert not high_f_limit < low_f_limit
        # print(f'low_f_limit: {low_f_limit} -- high_f_limit: {high_f_limit}')
        result = self._dfs_f(problem, problem.initial_state, f_limit=high_f_limit)
        assert result.is_solution_found

        search_result = SearchResult(
            solver=self,
            problem=problem,
            solution_path=GraphProblemStatesPath(result.solution_path),
            nr_expanded_states=total_nr_expanded_states,
            # nr_expanded_states_for_optimal_f_limit=result.nr_expanded_states,
            max_nr_stored_states=result.max_depth_reached,
            nr_iterations=nr_iterations)
        return search_result

    def _iterative_f_deepening(self, problem: GraphProblem) -> SearchResult:
        # Iterative deepening
        f_limit = 0
        total_nr_expanded_states = 0
        max_depth_reached = 0
        for nr_iterations in itertools.count(start=1):
            # print(f'iter: {nr_iterations} -- f_limit: {f_limit} -- max_depth_reached: {max_depth_reached} -- total_nr_expanded_states: {total_nr_expanded_states}')
            if self.max_nr_iterations is not None and nr_iterations > self.max_nr_iterations:
                return SearchResult(
                    solver=self,
                    problem=problem,
                    nr_expanded_states=total_nr_expanded_states,
                    max_nr_stored_states=max_depth_reached,
                    nr_iterations=nr_iterations-1,
                    stop_reason=StopReason.ExceededMaxNrIteration)

            max_nr_states_to_expand = None if self.max_nr_states_to_expand is None \
                else self.max_nr_states_to_expand - total_nr_expanded_states
            result = self._dfs_f(
                problem=problem, state=problem.initial_state, f_limit=f_limit,
                max_nr_states_to_expand=max_nr_states_to_expand)
            total_nr_expanded_states += result.nr_expanded_states
            max_depth_reached = max(max_depth_reached, result.max_depth_reached)

            if result.stopped_because_max_nr_states_to_expand_reached:
                return SearchResult(
                    solver=self,
                    problem=problem,
                    nr_expanded_states=total_nr_expanded_states,
                    max_nr_stored_states=max_depth_reached,
                    nr_iterations=nr_iterations,
                    stop_reason=StopReason.ExceededMaxNrStatesToExpand)

            if result.solution_path is not None:
                return SearchResult(
                    solver=self,
                    problem=problem,
                    solution_path=GraphProblemStatesPath(result.solution_path),
                    nr_expanded_states=total_nr_expanded_states,
                    max_nr_stored_states=max_depth_reached,
                    nr_iterations=nr_iterations)

            f_limit = result.f_limit_for_next_iteration
            assert f_limit is not None

    def _dfs_f(self, problem: GraphProblem, state: GraphProblemState, f_limit: float = 0,
               cumulative_cost: Optional[Cost] = None, last_operator_cost: Optional[Cost] = None,
               last_operator_name: Optional[str] = None, depth: int = 1,
               max_nr_states_to_expand: Optional[int] = None) -> DFSFResult:
        if cumulative_cost is None:
            cumulative_cost = problem.get_zero_cost()
        if last_operator_cost is None:
            last_operator_cost = problem.get_zero_cost()

        state_g = cumulative_cost.get_g_cost() if isinstance(cumulative_cost, ExtendedCost) else cumulative_cost
        state_h = self.heuristic_function.estimate(state)
        state_f = state_h * self.heuristic_weight + state_g * (1 - self.heuristic_weight)

        if state_f > f_limit:
            return DFSFResult(f_limit_for_next_iteration=state_f, max_depth_reached=depth)

        if problem.is_goal(state):
            assert math.isclose(state_h, 0)
            current_node = StatesPathNode(
                state=state, last_operator_cost=last_operator_cost, cumulative_cost=cumulative_cost,
                cumulative_g_cost=state_g, last_operator_name=last_operator_name)
            return DFSFResult(
                max_depth_reached=depth, solution_cost=cumulative_cost, solution_path=(current_node,))

        if max_nr_states_to_expand is not None and max_nr_states_to_expand < 1:
            return DFSFResult(max_depth_reached=depth, stopped_because_max_nr_states_to_expand_reached=True)

        f_limit_for_next_iteration = np.inf
        max_depth_reached = depth
        nr_expanded_states = 1

        for successor_idx, operator_result in enumerate(problem.expand_state_with_costs(state), start=1):
            assert isinstance(operator_result, OperatorResult)
            inner_call_result = self._dfs_f(
                problem=problem,
                state=operator_result.successor_state,
                f_limit=f_limit,
                cumulative_cost=cumulative_cost + operator_result.operator_cost,
                last_operator_cost=operator_result.operator_cost,
                last_operator_name=operator_result.operator_name,
                depth=depth + 1,
                max_nr_states_to_expand=None if max_nr_states_to_expand is None else max_nr_states_to_expand - nr_expanded_states)
            max_depth_reached = max(max_depth_reached, inner_call_result.max_depth_reached)
            nr_expanded_states += inner_call_result.nr_expanded_states
            if inner_call_result.stopped_because_max_nr_states_to_expand_reached:
                return DFSFResult(
                    nr_expanded_states=nr_expanded_states, max_depth_reached=max_depth_reached,
                    stopped_because_max_nr_states_to_expand_reached=True)
            if inner_call_result.solution_path is not None:
                curr_node_in_solution_path = StatesPathNode(
                    state=state, last_operator_cost=last_operator_cost, cumulative_cost=cumulative_cost,
                    cumulative_g_cost=state_g, last_operator_name=last_operator_name)
                solution_path = tuple(itertools.chain((curr_node_in_solution_path,), inner_call_result.solution_path))
                return DFSFResult(
                    max_depth_reached=max_depth_reached, nr_expanded_states=nr_expanded_states,
                    solution_cost=inner_call_result.solution_cost, solution_path=solution_path)
            assert inner_call_result.f_limit_for_next_iteration is not None
            if inner_call_result.f_limit_for_next_iteration > f_limit:
                f_limit_for_next_iteration = min(f_limit_for_next_iteration, inner_call_result.f_limit_for_next_iteration)

        return DFSFResult(f_limit_for_next_iteration=f_limit_for_next_iteration,
                          max_depth_reached=max_depth_reached,
                          nr_expanded_states=nr_expanded_states)
