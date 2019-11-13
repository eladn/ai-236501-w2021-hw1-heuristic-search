from .graph_problem_interface import *
from .best_first_search import BestFirstSearch
from typing import Optional
import numpy as np


class GreedyStochastic(BestFirstSearch):
    solver_name = 'GreedyStochastic'
    def __init__(self, heuristic_function_type: HeuristicFunctionType,
                 T_init: float = 1.0, N: int = 5, T_scale_factor: float = 0.95,
                 max_nr_states_to_expand: Optional[int] = None):
        # GreedyStochastic is a graph search algorithm. Hence, we use close set.
        super(GreedyStochastic, self).__init__(use_close=True, max_nr_states_to_expand=max_nr_states_to_expand)
        self.heuristic_function_type = heuristic_function_type
        self.T = T_init
        self.N = N
        self.T_scale_factor = T_scale_factor

    def _init_solver(self, problem: GraphProblem):
        super(GreedyStochastic, self)._init_solver(problem)
        self.heuristic_function = self.heuristic_function_type(problem)
        self.solver_name = f'{self.__class__.solver_name} (h={self.heuristic_function.heuristic_name})'

    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):
        """
        TODO: implement this method!
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

    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        """
        TODO: implement this method!
        Remember: `GreedyStochastic` is greedy.
        """

        # raise NotImplemented()  # TODO: remove!

        return self.heuristic_function.estimate(search_node.state)

    def _extract_next_search_node_to_expand(self, problem: GraphProblem) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         using the stochastic method to choose out of the N
         best items from open.
        TODO: implement this method!
        Use `np.random.choice(...)` whenever you need to randomly choose
         an item from an array of items given a probabilities array `p`.
        You can read the documentation of `np.random.choice(...)` and
         see usage examples by searching it in Google.
        Notice: You might want to pop min(N, len(open)) items from the
                `open` priority queue, and then choose an item out
                of these popped items. The other items have to be
                pushed again into that queue.
        """

        # raise NotImplemented()  # TODO: remove!

        if self.open.is_empty():
            return None

        num_nodes_to_pop = min(self.N, len(self.open))
        candidates = [self.open.pop_next_node() for _ in range(num_nodes_to_pop)]
        priorities = np.array([candidate.expanding_priority for candidate in candidates])
        if np.min(priorities) == 0:
            # Found goal
            idx_chosen = np.argmin(priorities)
        else:
            # Randomize next item:
            priorities = priorities / np.min(priorities)
            priorities = np.power(priorities, -1/self.T)
            probabilities = priorities / np.sum(priorities)
            idx_chosen = np.random.choice(num_nodes_to_pop, 1, replace=True, p=probabilities)[0]
        chosen_candidate = candidates.pop(idx_chosen)

        # Put the others (not chosen) back in the open queue.
        for node in candidates:
            self.open.push_node(node)

        self.close.add_node(chosen_candidate)

        self.T *= self.T_scale_factor  # Update T parameter

        return chosen_candidate
