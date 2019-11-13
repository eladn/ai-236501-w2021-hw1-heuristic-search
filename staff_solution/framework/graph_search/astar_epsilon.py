from .graph_problem_interface import *
from .astar import AStar
from typing import Optional, Callable
import numpy as np
import math


class AStarEpsilon(AStar):
    """
    This class implements the (weighted) A*Epsilon search algorithm.
    A*Epsilon algorithm basically works like the A* algorithm, but with
    another way to choose the next node to expand from the open queue.
    """

    solver_name = 'A*eps'

    def __init__(self,
                 heuristic_function_type: HeuristicFunctionType,
                 within_focal_priority_function: Callable[[SearchNode, GraphProblem, 'AStarEpsilon'], float],
                 heuristic_weight: float = 0.5,
                 focal_epsilon: float = 0.1,
                 max_nr_states_to_expand: Optional[int] = None,
                 max_focal_size: Optional[int] = None):
        """
        :param heuristic_function_type: The A* solver stores the constructor of the heuristic
                                        function, rather than an instance of that heuristic.
                                        In each call to "solve_problem" a heuristic instance
                                        is created.
        :param heuristic_weight: Used to calculate the f-score of a node using
                                 the heuristic value and the node's cost. Default is 0.5.
        """
        # A* is a graph search algorithm. Hence, we use close set.
        super(AStarEpsilon, self).__init__(heuristic_function_type, heuristic_weight,
                                           max_nr_states_to_expand=max_nr_states_to_expand)
        self.focal_epsilon = focal_epsilon
        if focal_epsilon < 0:
            raise ValueError(f'The argument `focal_epsilon` for A*eps should be >= 0; '
                             f'given focal_epsilon={focal_epsilon}.')
        # self.within_focal_priority_function_type = within_focal_priority_function_type
        # if within_focal_priority_function_type is None:
        #     self.within_focal_priority_function_type = self.heuristic_function_type
        # self.within_focal_priority_function = None
        self.within_focal_priority_function = within_focal_priority_function
        self.max_focal_size = max_focal_size

    def _init_solver(self, problem):
        super(AStarEpsilon, self)._init_solver(problem)
        # self.within_focal_priority_function = self.within_focal_priority_function_type(problem)

    def _extract_next_search_node_to_expand(self, problem: GraphProblem) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         by focusing on the current FOCAL and choosing the node
         with the best within_focal_priority from it.
        TODO: implement this method!
        Notice: You might want to pop items from the `open` priority queue,
                and then choose an item out of these popped items. Don't
                forget: the other items have to be pushed back into open.
        """

        # raise NotImplemented()  # TODO: remove!

        if self.open.is_empty():
            return None

        focal = []
        min_expanding_priority_in_open = self.open.peek_next_node().expanding_priority
        max_expanding_priority_in_focal = min_expanding_priority_in_open * (1 + self.focal_epsilon)
        while not self.open.is_empty() and \
                (self.open.peek_next_node().expanding_priority < max_expanding_priority_in_focal or
                 math.isclose(self.open.peek_next_node().expanding_priority, max_expanding_priority_in_focal)) and \
                (self.max_focal_size is None or len(focal) < self.max_focal_size):
            focal.append(self.open.pop_next_node())

        assert len(focal) > 0
        focal_priorities = np.array([
            self.within_focal_priority_function(candidate, problem, self)
            for candidate in focal
        ])
        idx_chosen = int(np.argmin(focal_priorities))
        chosen_candidate = focal.pop(idx_chosen)

        # Put the others (not chosen) back in the open queue.
        for node in focal:
            self.open.push_node(node)
        self.close.add_node(chosen_candidate)
        return chosen_candidate
