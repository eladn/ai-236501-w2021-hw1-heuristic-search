from .graph_problem_interface import *
from .astar import AStar
from typing import Optional, Callable
import numpy as np
import math
from .utils.timer import Timer


class AnytimeAStar(GraphProblemSolver):
    """
    This class implements the (weighted) anytime A* search algorithm.
    TODO: explain here
    """

    solver_name = 'Anytime-A*'

    def __init__(self,
                 heuristic_function_type: HeuristicFunctionType,
                 max_nr_states_to_expand_per_iteration: int):
        self.heuristic_function_type = heuristic_function_type
        self.max_nr_states_to_expand_per_iteration = max_nr_states_to_expand_per_iteration

    def solve_problem(self, problem: GraphProblem) -> SearchResult:
        with Timer(print_title=False) as timer:
            total_nr_expanded_states = 0

            acceptable_astar = AStar(heuristic_function_type=self.heuristic_function_type, heuristic_weight=0.5,
                                     max_nr_states_to_expand=self.max_nr_states_to_expand_per_iteration)
            acceptable_astar_res = acceptable_astar.solve_problem(problem)
            total_nr_expanded_states += acceptable_astar_res.nr_expanded_states
            if acceptable_astar_res.is_solution_found:
                return acceptable_astar_res._replace(
                    solver=self, nr_expanded_states=total_nr_expanded_states, solving_time=timer.elapsed)  #, 0.5

            greedy = AStar(heuristic_function_type=self.heuristic_function_type, heuristic_weight=1,
                           max_nr_states_to_expand=self.max_nr_states_to_expand_per_iteration)
            greedy_res = greedy.solve_problem(problem)
            total_nr_expanded_states += greedy_res.nr_expanded_states
            if not greedy_res.is_solution_found:
                return greedy_res._replace(
                    solver=self, nr_expanded_states=total_nr_expanded_states, solving_time=timer.elapsed)  # 1
            best_solution = greedy_res

            high_heuristic_weight = 1.0
            low_heuristic_weight = 0.5
            while (high_heuristic_weight - low_heuristic_weight) > 0.01:
                mid_heuristic_weight = (low_heuristic_weight + high_heuristic_weight) / 2
                print(f'low: {low_heuristic_weight} -- mid: {mid_heuristic_weight} -- high: {high_heuristic_weight}')
                astar = AStar(heuristic_function_type=self.heuristic_function_type,
                              heuristic_weight=mid_heuristic_weight,
                              max_nr_states_to_expand=self.max_nr_states_to_expand_per_iteration)
                res = astar.solve_problem(problem)
                total_nr_expanded_states += res.nr_expanded_states
                if res.is_solution_found:
                    high_heuristic_weight = mid_heuristic_weight
                    best_solution = res if res.solution_g_cost < best_solution.solution_g_cost else best_solution
                else:
                    low_heuristic_weight = mid_heuristic_weight
        return best_solution._replace(
            solver=self, nr_expanded_states=total_nr_expanded_states, solving_time=timer.elapsed)  #, high_heuristic_weight
