from .graph_problem_interface import *
from .greedy_stochastic import GreedyStochastic
from .utils.utils import calc_relative_error
from typing import Optional, Tuple
from dataclasses import dataclass
import itertools
import math
import numpy as np


@dataclass(frozen=True)
class DFSLResult:
    f_limit: float
    max_depth_reached: int
    nr_expanded_states: int = 0
    solution_cost: Optional[Cost] = None
    solution_path: Optional[Tuple[StatesPathNode, ...]] = None

    @property
    def is_solution_found(self):
        return self.solution_path is not None


class IDAStar(GraphProblemSolver):
    solver_name: str = 'ID-A*'

    def __init__(self, heuristic_function_type: HeuristicFunctionType, heuristic_weight: float = 0.5, max_cost_relative_error: float = 0.005):
        self.heuristic_function_type = heuristic_function_type
        self.heuristic_function = None
        self.heuristic_weight = heuristic_weight
        self.max_cost_relative_error = max_cost_relative_error
        self.solver_name += ' (h={heuristic_name}, w={heuristic_weight:.3f})'.format(
            heuristic_name=heuristic_function_type.heuristic_name if hasattr(heuristic_function_type,
                                                                             'heuristic_name') else 'UnknownHeuristic',
            heuristic_weight=self.heuristic_weight)

    def solve_problem(self, problem: GraphProblem) -> SearchResult:
        self.heuristic_function = self.heuristic_function_type(problem)
        return self._binary_search_f_deepening(problem)

    def _binary_search_f_deepening(self, problem: GraphProblem) -> SearchResult:
        # Ensure that there is no solution for f_limit=0 (in order to to use it as `low_f_limit`)
        zero_f_limit_result = self._dfs_f(problem, problem.initial_state, f_limit=0)
        if zero_f_limit_result.solution_path is not None:
            return SearchResult(
                solver=self,
                problem=problem,
                solution_path=GraphProblemStatesPath(zero_f_limit_result.solution_path),
                nr_expanded_states=zero_f_limit_result.nr_expanded_states,
                # nr_expanded_states_for_optimal_f_limit=result.nr_expanded_states,
                max_nr_stored_states=zero_f_limit_result.max_depth_reached,
                solving_time=0,
                nr_iterations=1)

        # Quickly find some some f_limit for which a solution exists (in order to to use it as `high_f_limit`)
        greedy = GreedyStochastic(self.heuristic_function_type)
        greedy_solution = greedy.solve_problem(problem)
        assert greedy_solution.is_solution_found
        greedy_solution_cost = greedy_solution.solution_g_cost
        assert greedy_solution_cost > 0

        # Binary deepening
        low_f_limit = 0  # Not found solution at this f_limit
        high_f_limit = greedy_solution_cost * (1 - self.heuristic_weight)  # Found solution at this f_limit

        nr_iterations = 1
        nr_expanded_states = 0
        max_f_limit_relative_error = self.max_cost_relative_error * (1 - self.heuristic_weight)
        while calc_relative_error(low_f_limit, high_f_limit) > max_f_limit_relative_error:
            assert high_f_limit > low_f_limit
            mid_f_limit = (high_f_limit + low_f_limit) / 2
            # print(f'low_f_limit: {low_f_limit} -- mid_f_limit: {mid_f_limit} -- high_f_limit: {high_f_limit}')
            result = self._dfs_f(problem, problem.initial_state, f_limit=mid_f_limit)
            nr_expanded_states += result.nr_expanded_states
            if result.is_solution_found:
                high_f_limit = mid_f_limit
            else:
                low_f_limit = mid_f_limit
            nr_iterations += 1

        assert not high_f_limit < low_f_limit
        # print(f'low_f_limit: {low_f_limit} -- high_f_limit: {high_f_limit}')
        result = self._dfs_f(problem, problem.initial_state, f_limit=high_f_limit)
        assert result.is_solution_found

        search_result = SearchResult(
            solver=self,
            problem=problem,
            solution_path=GraphProblemStatesPath(result.solution_path),
            nr_expanded_states=nr_expanded_states,
            # nr_expanded_states_for_optimal_f_limit=result.nr_expanded_states,
            max_nr_stored_states=result.max_depth_reached,
            solving_time=0,
            nr_iterations=nr_iterations)
        return search_result

    def _iterative_f_deepening(self, problem: GraphProblem) -> SearchResult:
        # Iterative deepening
        f_limit = 0
        while True:
            result = self._dfs_f(problem, problem.initial_state, f_limit=f_limit)
            f_limit = result.f_limit
            print(f'f_limit: {f_limit}')
            if result.solution_path is not None:
                print(result)
                search_result = SearchResult(
                    solver=self,
                    problem=problem,
                    solution_path=GraphProblemStatesPath(result.solution_path),
                    nr_expanded_states=result.nr_expanded_states,
                    max_nr_stored_states=result.max_depth_reached,
                    solving_time=0)
                return search_result

    def _dfs_f(self, problem: GraphProblem, state: GraphProblemState, f_limit: float = 0,
               cumulative_cost: Optional[Cost] = None, last_operator_cost: Optional[Cost] = None,
               last_operator_name: Optional[str] = None, depth: int = 1) -> DFSLResult:
        if cumulative_cost is None:
            cumulative_cost = problem.get_zero_cost()
        if last_operator_cost is None:
            last_operator_cost = problem.get_zero_cost()

        state_g = cumulative_cost.get_g_cost() if isinstance(cumulative_cost, ExtendedCost) else cumulative_cost
        state_h = self.heuristic_function.estimate(state)
        state_f = state_h * self.heuristic_weight + state_g * (1 - self.heuristic_weight)
        if state_f > f_limit:
            return DFSLResult(f_limit=state_f, max_depth_reached=depth)
        if problem.is_goal(state):
            assert math.isclose(state_h, 0)
            # assert self.heuristic_weight > 0.5 or f_limit == state_f
            return DFSLResult(
                f_limit=f_limit, max_depth_reached=depth, solution_cost=cumulative_cost, solution_path=(StatesPathNode(
                    state=state, last_operator_cost=last_operator_cost, cumulative_cost=cumulative_cost,
                    cumulative_g_cost=state_g, last_operator_name=last_operator_name),))
        next_f_limit = np.inf
        max_depth_reached = depth
        nr_expanded_states = 0

        for successor_idx, operator_result in enumerate(problem.expand_state_with_costs(state), start=1):
            assert isinstance(operator_result, OperatorResult)
            inner_call_result = self._dfs_f(
                problem=problem,
                state=operator_result.successor_state,
                f_limit=f_limit,
                cumulative_cost=cumulative_cost + operator_result.operator_cost,
                last_operator_cost=operator_result.operator_cost,
                last_operator_name=operator_result.operator_name,
                depth=depth + 1)
            max_depth_reached = max(max_depth_reached, inner_call_result.max_depth_reached)
            nr_expanded_states += 1 + inner_call_result.nr_expanded_states
            if inner_call_result.solution_path is not None:
                current_node = StatesPathNode(
                    state=state, last_operator_cost=last_operator_cost, cumulative_cost=cumulative_cost,
                    cumulative_g_cost=state_g, last_operator_name=last_operator_name)
                return DFSLResult(
                    f_limit=f_limit, max_depth_reached=max_depth_reached,
                    nr_expanded_states=nr_expanded_states, solution_cost=inner_call_result.solution_cost,
                    solution_path=tuple(itertools.chain((current_node,), inner_call_result.solution_path)))
            if inner_call_result.f_limit > f_limit:
                next_f_limit = min(next_f_limit, inner_call_result.f_limit)

        return DFSLResult(f_limit=next_f_limit, max_depth_reached=max_depth_reached,
                          nr_expanded_states=nr_expanded_states)
