from .graph_problem_interface import *
from .best_first_search import BestFirstSearch
from typing import Optional


class GreedyBestFirst(BestFirstSearch):
    """
    This class implements the Weighted-A* search algorithm.
    A* algorithm is in the Best First Search algorithms family.
    """

    solver_name = 'Greedy'

    def __init__(self, heuristic_function_type: HeuristicFunctionType,
                 max_nr_states_to_expand: Optional[int] = None):
        """
        :param heuristic_function_type: The greedy solver stores the constructor of the heuristic
                                        function, rather than an instance of that heuristic.
                                        In each call to "solve_problem" a heuristic instance
                                        is created.
        """
        # A* is a graph search algorithm. Hence, we use close set.
        super(GreedyBestFirst, self).__init__(use_close=True, max_nr_states_to_expand=max_nr_states_to_expand)
        self.heuristic_function_type = heuristic_function_type
        self.heuristic_function = None

    def _init_solver(self, problem):
        """
        Called by "solve_problem()" in the implementation of `BestFirstSearch`.
        The problem to solve is known now, so we can create the heuristic function to be used.
        """
        super(GreedyBestFirst, self)._init_solver(problem)
        self.heuristic_function = self.heuristic_function_type(problem)
        self.solver_name = f'{self.__class__.solver_name} (h={self.heuristic_function.heuristic_name})'

    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        """
        Called by solve_problem() in the implementation of `BestFirstSearch`
         whenever just after creating a new successor node.
        Should calculate and return the f-score of the given node.
        This score is used as a priority of this node in the open priority queue.

        TODO: implement this method.
        Remember: The Greedy-Best-First is a Best-First-Search algorithm in which f = h.
        """

        # raise NotImplemented()  # TODO: remove!

        return self.heuristic_function.estimate(search_node.state)

    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):
        """
        Called by solve_problem() in the implementation of `BestFirstSearch`
         whenever creating a new successor node.
        This method is responsible for adding this just-created successor
         node into the `self.open` priority queue, and may check the existence
         of another node representing the same state in `self.close`.

        TODO: implement this method.
        Have a look at the implementation of `BestFirstSearch` to have better understanding.
        Use `self.open` (SearchNodesPriorityQueue) and `self.close` (SearchNodesCollection) data structures.
        These data structures are implemented in `graph_search/best_first_search.py`.
        Note: At this stage, the successor_node's f-score has been already calculated and stored under
              `successor_node.expanding_priority`.
        """

        # raise NotImplemented()  # TODO: remove!

        # A neighbour might be already closed, but still could be improved since we use a greedy method.
        if self.close.has_state(successor_node.state):
            already_closed_node_with_same_state = self.close.get_node_by_state(successor_node.state)
            assert already_closed_node_with_same_state is not None
            if already_closed_node_with_same_state.expanding_priority <= successor_node.expanding_priority:
                return
            self.close.remove_node(already_closed_node_with_same_state)

        if self.open.has_state(successor_node.state):
            already_found_node_with_same_state = self.open.get_node_by_state(successor_node.state)
            if already_found_node_with_same_state.expanding_priority > successor_node.expanding_priority:
                self.open.extract_node(already_found_node_with_same_state)

        if not self.open.has_state(successor_node.state):
            self.open.push_node(successor_node)
