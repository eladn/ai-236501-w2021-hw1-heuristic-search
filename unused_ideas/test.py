from framework import *
from deliveries import *
from main import run_astar_for_weights_in_range

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union

# Load the map
streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path("tlv_streets_map.csv"))

# Make `np.random` behave deterministic.
Consts.set_seed()


def packages_truck_deliveries_problem():
    print()
    print('Solve the packages truck deliveries problem.')

    problem_inputs_by_size = {}

    def get_deliveries_problem(problem_input_size: str = 'small', optimization_objective: OptimizationObjective = OptimizationObjective.Distance):
        assert problem_input_size in {'small', 'moderate', 'big'}
        if problem_input_size not in problem_inputs_by_size:
            problem_inputs_by_size[problem_input_size] = DeliveriesTruckProblemInput.load_from_file(
                f'packages_truck_deliveries_{problem_input_size}_input.in', streets_map)
        return DeliveriesTruckProblem(
            problem_input=problem_inputs_by_size[problem_input_size],
            streets_map=streets_map,
            optimization_objective=optimization_objective)

    # for w in (0.8, 0.7, 0.65, 0.6, 0.5, 0.35):
    #     astar = AStar(deliveries_heuristic_type, heuristic_weight=w)
    #     res = astar.solve_problem(small_deliveries_problem)
    #     print(res)

    # for w in (0.8, 0.7, 0.65, 0.6, 0.5):
    #     astar = AStar(deliveries_heuristic_type, heuristic_weight=w)
    #     res = astar.solve_problem(moderate_deliveries_problem)
    #     print(res)

    # for w in (1, 0.9, 0.86):
    #     astar = AStar(deliveries_heuristic_type, heuristic_weight=w)
    #     res = astar.solve_problem(big_deliveries_problem)
    #     print(res)

    for w in (0.65,):
        # uc = UniformCost()
        # res = uc.solve_problem(get_deliveries_problem('small', OptimizationObjective.Distance))
        # print(res)

        # astar = AStar(TruckDeliveriesMaxAirDistHeuristic, heuristic_weight=w)
        # res = astar.solve_problem(get_deliveries_problem('small', OptimizationObjective.Distance))
        # print(res)
        # astar = AStar(TruckDeliveriesMaxAirDistHeuristic, heuristic_weight=w)
        # res = astar.solve_problem(get_deliveries_problem('small', OptimizationObjective.Time))
        # print(res)
        # astar = AStar(TruckDeliveriesMaxAirDistHeuristic, heuristic_weight=w)
        # res = astar.solve_problem(get_deliveries_problem('small', OptimizationObjective.Money))
        # print(res)

        run_astar_for_weights_in_range(TruckDeliveriesMSTAirDistHeuristic, get_deliveries_problem('small', OptimizationObjective.Distance))

        astar = AStar(TruckDeliveriesMSTAirDistHeuristic, heuristic_weight=w)
        res = astar.solve_problem(get_deliveries_problem('moderate', OptimizationObjective.Distance))
        print(res)
        # astar = AStar(TruckDeliveriesMSTAirDistHeuristic, heuristic_weight=w)
        # res = astar.solve_problem(get_deliveries_problem('small', OptimizationObjective.Time))
        # print(res)
        # astar = AStar(TruckDeliveriesMSTAirDistHeuristic, heuristic_weight=w)
        # res = astar.solve_problem(get_deliveries_problem('small', OptimizationObjective.Money))
        # print(res)

        # problem_time = get_deliveries_problem('moderate', OptimizationObjective.Time)
        # problem_money = get_deliveries_problem('moderate', OptimizationObjective.Money)
        # focal_heuristic = TruckDeliveriesMSTAirDistHeuristic(problem_money)
        # def focal_priority_fn(node: SearchNode) -> float:
        #     if not hasattr(node, '__within_focal_priority'):
        #         within_focal_priority = node.cost.money_cost + focal_heuristic.estimate(node.state)
        #         setattr(node, '__within_focal_priority', within_focal_priority)
        #     return getattr(node, '__within_focal_priority')
        # astar_eps = AStarEpsilon(TruckDeliveriesMSTAirDistHeuristic, focal_priority_fn, heuristic_weight=w, focal_epsilon=0.1,
        #                          max_nr_states_to_expand=40000)
        # res = astar_eps.solve_problem(problem_time)
        # print(res)

        # Try to see if A*eps can help reduce #dev.
        # astar = AStar(TruckDeliveriesMSTAirDistHeuristic, heuristic_weight=0.5)
        # res = astar.solve_problem(get_deliveries_problem('moderate', OptimizationObjective.Time))
        # print(res)
        # within_focal_priority_function = lambda node: node.g_cost
        # astar_eps = AStarEpsilon(
        #     TruckDeliveriesMSTAirDistHeuristic, within_focal_priority_function=within_focal_priority_function,
        #     heuristic_weight=0.5, max_nr_states_to_expand=30000)
        # res = astar_eps.solve_problem(get_deliveries_problem('small', OptimizationObjective.Time))
        # print(res)
        # within_focal_priority_function = lambda node, problem, solver: node.expanding_priority - (1 - solver.heuristic_weight) * node.g_cost
        # astar_eps = AStarEpsilon(
        #     TruckDeliveriesMSTAirDistHeuristic, within_focal_priority_function=within_focal_priority_function,
        #     heuristic_weight=0.5, max_nr_states_to_expand=40000, max_focal_size=25)
        # res = astar_eps.solve_problem(get_deliveries_problem('moderate', OptimizationObjective.Time))
        # print(res)

        # astar = AStar(TruckDeliveriesMSTAirDistHeuristic, heuristic_weight=w)
        # res = astar.solve_problem(get_deliveries_problem('small', OptimizationObjective.Distance))
        # print(res)
        # greedy = GreedyBestFirst(TruckDeliveriesMSTAirDistHeuristic)
        # res = greedy.solve_problem(get_deliveries_problem('small', OptimizationObjective.Distance))
        # print(res)
        # greedy_stochastic = GreedyStochastic(TruckDeliveriesMSTAirDistHeuristic)
        # res = greedy_stochastic.solve_problem(get_deliveries_problem('small', OptimizationObjective.Distance))
        # print(res)

        # astar = AStar(TruckDeliveriesMSTAirDistHeuristic, heuristic_weight=w)
        # res = astar.solve_problem(get_deliveries_problem('moderate', OptimizationObjective.Time))
        # print(res)
        # astar = AStar(TruckDeliveriesMSTAirDistHeuristic, heuristic_weight=w)
        # res = astar.solve_problem(get_deliveries_problem('small', OptimizationObjective.Money))
        # print(res)
        # id_astar = IDAStar(TruckDeliveriesMSTAirDistHeuristic, heuristic_weight=w, deepening_technique='iterative',
        #                    max_nr_iterations=10, max_nr_states_to_expand=40000)
        # res = id_astar.solve_problem(get_deliveries_problem('small'))
        # print(res)
        # id_astar = IDAStar(TruckDeliveriesMSTAirDistHeuristic, heuristic_weight=w, deepening_technique='iterative',
        #                    max_nr_iterations=1000, max_nr_states_to_expand=40000)
        # res = id_astar.solve_problem(get_deliveries_problem('small'))
        # print(res)
        # id_astar = IDAStar(TruckDeliveriesMSTAirDistHeuristic, heuristic_weight=w, deepening_technique='binary_search')
        # res = id_astar.solve_problem(get_deliveries_problem('small'))
        # print(res)

    # for _ in range(10):
    #     greedy = GreedyStochastic(deliveries_heuristic_type)
    #     res = greedy.solve_problem(moderate_deliveries_problem)
    #     print(res)


if __name__ == '__main__':
    # multiple_agents_pizza_deliveries_problem()
    packages_truck_deliveries_problem()
