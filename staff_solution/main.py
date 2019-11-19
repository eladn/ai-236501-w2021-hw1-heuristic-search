from framework import *
from deliveries import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union, Optional

# Load the streets map
streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path("tlv_streets_map.csv"))

# Make sure that the whole execution is deterministic.
# This is important, because we expect to get the exact same results
# in each execution.
Consts.set_seed()


# --------------------------------------------------------------------
# ------------------------ StreetsMap Problem ------------------------
# --------------------------------------------------------------------

def plot_distance_and_expanded_wrt_weight_figure(
        problem_name: str,
        weights: Union[np.ndarray, List[float]],
        total_cost: Union[np.ndarray, List[float]],
        total_nr_expanded: Union[np.ndarray, List[int]]):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    TODO [Ex.14]: Complete the implementation of this method.
    """
    weights, total_cost, total_nr_expanded = np.array(weights), np.array(total_cost), np.array(total_nr_expanded)
    assert len(weights) == len(total_cost) == len(total_nr_expanded)
    assert len(weights) > 0
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert is_sorted(weights)

    fig, ax1 = plt.subplots()

    # TODO: Plot the total distances with ax1. Use `ax1.plot(...)`.
    # TODO: Make this curve colored blue with solid line style.
    # See documentation here:
    # https://matplotlib.org/2.0.0/api/_as_gen/matplotlib.axes.Axes.plot.html
    # You can also Google for additional examples.
    # raise NotImplemented()  # TODO: remove this line!
    p1, = ax1.plot(weights, total_cost, 'b-', label='Solution cost')

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('solution cost', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()

    # TODO: Plot the total expanded with ax2. Use `ax2.plot(...)`.
    # TODO: ax2: Make the y-axis label, ticks and tick labels match the line color.
    # TODO: Make this curve colored red with solid line style.
    # raise NotImplemented()  # TODO: remove this line!
    p2, = ax2.plot(weights, total_nr_expanded, 'r-', label='#Expanded states')
    ax2.set_ylabel('states expanded', color='r')
    ax2.tick_params('y', colors='r')

    curves = [p1, p2]
    ax1.legend(curves, [curve.get_label() for curve in curves])

    fig.tight_layout()
    plt.title(f'Quality vs. time for wA* \non problem {problem_name}')
    plt.show()


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem, n: int = 30,
                                   max_nr_states_to_expand: Optional[int] = 30_000):
    # TODO [Ex.14]:
    #  1. Create an array of 20 numbers equally spread in [0.5, 1]
    #     (including the edges). You can use `np.linspace()` for that.
    #  2. For each weight in that array run the wA* algorithm, with the
    #     given `heuristic_type` over the given problem. For each such run,
    #     if a solution has been found (res.is_solution_found), store the
    #     cost of the solution (res.solution_g_cost), the number of
    #     expanded states (res.nr_expanded_states), and the weight that
    #     has been used in this iteration. Store these in 3 lists (list
    #     for the costs, list for the #expanded and list for the weights).
    #     These lists should be of the same size when this operation ends.
    #     Don't forget to pass `max_nr_states_to_expand` to the AStar c'tor.
    #  3. Call the function `plot_distance_and_expanded_wrt_weight_figure()`
    #     with these 3 generated lists.
    # raise NotImplemented()  # TODO: remove!
    total_cost = []
    total_nr_expanded = []
    weights = np.linspace(0.5, 1, n)
    weights_with_found_solutions = []
    for w in weights:
        astar = AStar(heuristic_type, w, max_nr_states_to_expand=max_nr_states_to_expand)
        res = astar.solve_problem(problem)
        print(res)
        if res.is_solution_found:
            total_cost.append(res.solution_g_cost)
            total_nr_expanded.append(res.nr_expanded_states)
            weights_with_found_solutions.append(w)
    plot_distance_and_expanded_wrt_weight_figure(problem.name, weights_with_found_solutions, total_cost, total_nr_expanded)


def toy_map_problem_experiments():
    print()
    print('Solve the map problem.')

    # Ex.10
    # TODO: Just run it and inspect the printed result.
    toy_map_problem = MapProblem(streets_map, 54, 549)
    uc = UniformCost()
    res = uc.solve_problem(toy_map_problem)
    print(res)

    # Ex.12
    # TODO: create an instance of `AStar` with the `NullHeuristic`,
    #       solve the same `toy_map_problem` with it and print the results (as before).
    # Notice: AStar constructor receives the heuristic *type* (ex: `MyHeuristicClass`),
    #         and NOT an instance of the heuristic (eg: not `MyHeuristicClass()`).
    # exit()  # TODO: remove!
    astar = AStar(NullHeuristic)
    res = astar.solve_problem(toy_map_problem)
    print(res)

    # Ex.13
    # TODO: create an instance of `AStar` with the `AirDistHeuristic`,
    #       solve the same `toy_map_problem` with it and print the results (as before).
    # exit()  # TODO: remove!
    astar = AStar(AirDistHeuristic)
    res = astar.solve_problem(toy_map_problem)
    print(res)

    # Ex.14
    # TODO:
    #  1. Complete the implementation of the function
    #     `run_astar_for_weights_in_range()` (upper in this file).
    #  2. Complete the implementation of the function
    #     `plot_distance_and_expanded_by_weight_figure()`
    #     (upper in this file).
    #  3. Call here the function `run_astar_for_weights_in_range()`
    #     with `AirDistHeuristic` and `toy_map_problem`.
    # exit()  # TODO: remove!
    run_astar_for_weights_in_range(AirDistHeuristic, toy_map_problem)


# --------------------------------------------------------------------
# --------------------- Truck Deliveries Problem ---------------------
# --------------------------------------------------------------------

loaded_problem_inputs_by_size = {}


def get_deliveries_problem(problem_input_size: str = 'small', optimization_objective: OptimizationObjective = OptimizationObjective.Distance):
    assert problem_input_size in {'small', 'moderate', 'big'}
    if problem_input_size not in loaded_problem_inputs_by_size:
        loaded_problem_inputs_by_size[problem_input_size] = DeliveriesTruckProblemInput.load_from_file(
            f'{problem_input_size}_delivery.in', streets_map)
    return DeliveriesTruckProblem(
        problem_input=loaded_problem_inputs_by_size[problem_input_size],
        streets_map=streets_map,
        optimization_objective=optimization_objective)


def basic_deliveries_truck_problem_experiments():
    print()
    print('Solve the truck deliveries problem (small input, only distance objective, UniformCost).')

    small_delivery_problem_with_distance_cost = get_deliveries_problem('small', OptimizationObjective.Distance)

    # Ex.16
    # TODO: create an instance of `UniformCost`, solve the `small_delivery_problem_with_distance_cost`
    #       with it and print the results.
    # exit()  # TODO: remove!
    uniform_cost = UniformCost()
    res = uniform_cost.solve_problem(small_delivery_problem_with_distance_cost)
    print(res)


def deliveries_truck_problem_with_astar_experiments():
    print()
    print('Solve the truck deliveries problem (moderate input, only distance objective, A*, MaxAirDist & SumAirDist & MSTAirDist heuristics).')

    moderate_delivery_problem_with_distance_cost = get_deliveries_problem('moderate', OptimizationObjective.Distance)

    # Ex.18
    # TODO: create an instance of `AStar` with the `TruckDeliveriesMaxAirDistHeuristic`,
    #       solve the `moderate_delivery_problem_with_distance_cost` with it and print the results.
    # exit()  # TODO: remove!
    astar = AStar(TruckDeliveriesMaxAirDistHeuristic)
    res = astar.solve_problem(moderate_delivery_problem_with_distance_cost)
    print(res)

    # Ex.21
    # TODO: create an instance of `AStar` with the `TruckDeliveriesSumAirDistHeuristic`,
    #       solve the `moderate_delivery_problem_with_distance_cost` with it and print the results.
    # exit()  # TODO: remove!
    astar = AStar(TruckDeliveriesSumAirDistHeuristic)
    res = astar.solve_problem(moderate_delivery_problem_with_distance_cost)
    print(res)

    # Ex.24
    # TODO: create an instance of `AStar` with the `TruckDeliveriesMSTAirDistHeuristic`,
    #       solve the `moderate_delivery_problem_with_distance_cost` with it and print the results.
    # exit()  # TODO: remove!
    astar = AStar(TruckDeliveriesMSTAirDistHeuristic)
    res = astar.solve_problem(moderate_delivery_problem_with_distance_cost)
    print(res)


def deliveries_truck_problem_with_weighted_astar_experiments():
    print()
    print('Solve the truck deliveries problem (small & moderate input, only distance objective, wA*).')

    small_delivery_problem_with_distance_cost = get_deliveries_problem('small', OptimizationObjective.Distance)
    moderate_delivery_problem_with_distance_cost = get_deliveries_problem('moderate', OptimizationObjective.Distance)

    # Ex.26
    # TODO: Call here the function `run_astar_for_weights_in_range()`
    #       with `TruckDeliveriesMSTAirDistHeuristic`
    #       over the `small_delivery_problem_with_distance_cost`.
    # exit()  # TODO: remove!
    run_astar_for_weights_in_range(TruckDeliveriesMSTAirDistHeuristic, small_delivery_problem_with_distance_cost)

    # Ex.26
    # TODO: Call here the function `run_astar_for_weights_in_range()`
    #       with `TruckDeliveriesSumAirDistHeuristic`
    #       over the `moderate_delivery_problem_with_distance_cost`.
    # exit()  # TODO: remove!
    run_astar_for_weights_in_range(TruckDeliveriesSumAirDistHeuristic, moderate_delivery_problem_with_distance_cost)


def multiple_objectives_deliveries_truck_problem_experiments():
    print()
    print('Solve the truck deliveries problem (small input, time & money objectives).')

    small_delivery_problem_with_time_cost = get_deliveries_problem('small', OptimizationObjective.Time)
    small_delivery_problem_with_money_cost = get_deliveries_problem('small', OptimizationObjective.Money)

    # Ex.29
    # TODO: create an instance of `AStar` with the `TruckDeliveriesMSTAirDistHeuristic`,
    #       solve the `small_delivery_problem_with_time_cost` with it and print the results.
    # exit()  # TODO: remove!
    astar = AStar(TruckDeliveriesMSTAirDistHeuristic)
    res = astar.solve_problem(small_delivery_problem_with_time_cost)
    print(res)

    # Ex.29
    # TODO: create an instance of `AStar` with the `TruckDeliveriesMSTAirDistHeuristic`,
    #       solve the `small_delivery_problem_with_money_cost` with it and print the results.
    # exit()  # TODO: remove!
    astar = AStar(TruckDeliveriesMSTAirDistHeuristic)
    res = astar.solve_problem(small_delivery_problem_with_money_cost)
    print(res)


def deliveries_truck_problem_with_astar_epsilon_experiments():
    print()
    print('Solve the truck deliveries problem (small input, distance objective, using A*eps).')

    small_delivery_problem_with_distance_cost = get_deliveries_problem('small', OptimizationObjective.Distance)
    moderate_delivery_problem_with_distance_cost = get_deliveries_problem('moderate', OptimizationObjective.Distance)

    # Ex.xxx
    # Try using A*eps to improve the speed (#dev) for heuristic_weight=0.5.
    # TODO: create an instance of `AStarEpsilon` with the `TruckDeliveriesMSTAirDistHeuristic`,
    #       solve the `small_delivery_problem_with_distance_cost` with it and print the results.
    # exit()  # TODO: remove!

    def within_focal_h_priority_function(node: SearchNode, problem: GraphProblem, solver: AStarEpsilon):
        return node.expanding_priority - (1 - solver.heuristic_weight) * node.g_cost

    astar_eps = AStarEpsilon(
        TruckDeliveriesMSTAirDistHeuristic, within_focal_priority_function=within_focal_h_priority_function,
        max_nr_states_to_expand=8_000, max_focal_size=25)
    res = astar_eps.solve_problem(small_delivery_problem_with_distance_cost)
    print(res)

    # Ex.xxx
    # Try using A*eps to improve the solution quality when heuristic_weight>0.5.
    # TODO: create an instance of `AStarEpsilon` with the `TruckDeliveriesMSTAirDistHeuristic`,
    #       solve the `small_delivery_problem_with_distance_cost` with it and print the results.
    # exit()  # TODO: remove!

    def within_focal_g_priority_function(node: SearchNode, problem: GraphProblem, solver: AStarEpsilon):
        return node.g_cost

    astar07 = AStar(TruckDeliveriesMSTAirDistHeuristic, heuristic_weight=0.7)
    res = astar07.solve_problem(small_delivery_problem_with_distance_cost)
    print(res)
    astar_eps = AStarEpsilon(
        TruckDeliveriesMSTAirDistHeuristic, within_focal_priority_function=within_focal_g_priority_function,
        max_nr_states_to_expand=8_000, max_focal_size=20, heuristic_weight=0.7)
    res = astar_eps.solve_problem(small_delivery_problem_with_distance_cost)
    print(res)

    astar07 = AStar(TruckDeliveriesMSTAirDistHeuristic, heuristic_weight=0.7)
    res = astar07.solve_problem(moderate_delivery_problem_with_distance_cost)
    print(res)
    astar_eps = AStarEpsilon(
        TruckDeliveriesMSTAirDistHeuristic, within_focal_priority_function=within_focal_g_priority_function,
        max_nr_states_to_expand=8_000, max_focal_size=20, heuristic_weight=0.7)
    res = astar_eps.solve_problem(moderate_delivery_problem_with_distance_cost)
    print(res)


def deliveries_truck_problem_anytime_astar_experiments():
    print()
    print('Solve the truck deliveries problem (moderate input, only distance objective, Anytime-A*, '
          'MSTAirDist heuristics).')

    moderate_delivery_problem_with_distance_cost = get_deliveries_problem('moderate', OptimizationObjective.Distance)

    # Ex.xxx
    # TODO: create an instance of `AnytimeAStar` once with the `TruckDeliveriesMSTAirDistHeuristic`, with
    #       `max_nr_states_to_expand_per_iteration` set to 50, solve the
    #       `moderate_delivery_problem_with_distance_cost` with it and print the results.
    # exit()  # TODO: remove!
    anytime_astar = AnytimeAStar(TruckDeliveriesMSTAirDistHeuristic, max_nr_states_to_expand_per_iteration=50)
    res = anytime_astar.solve_problem(moderate_delivery_problem_with_distance_cost)
    print(res)


def big_deliveries_truck_problem_with_non_acceptable_heuristic_and_anytime_astar_experiments():
    print()
    print('Solve the truck deliveries problem (big input, only distance objective, Anytime-A*, '
          'SumAirDist & MSTAirDist heuristics).')

    big_delivery_problem_with_distance_cost = get_deliveries_problem('big', OptimizationObjective.Distance)

    # Ex.xxx
    # TODO: create an instance of `AnytimeAStar` once with the `TruckDeliveriesSumAirDistHeuristic`,
    #       and then with the `TruckDeliveriesMSTAirDistHeuristic`, both with `max_nr_states_to_expand_per_iteration`
    #       set to 400, solve the `big_delivery_problem_with_distance_cost` with it and print the results.
    # exit()  # TODO: remove!

    anytime_astar = AnytimeAStar(TruckDeliveriesSumAirDistHeuristic, max_nr_states_to_expand_per_iteration=400)
    res = anytime_astar.solve_problem(big_delivery_problem_with_distance_cost)
    print(res)

    anytime_astar = AnytimeAStar(TruckDeliveriesMSTAirDistHeuristic, max_nr_states_to_expand_per_iteration=400)
    res = anytime_astar.solve_problem(big_delivery_problem_with_distance_cost)
    print(res)


def deliveries_truck_problem_with_id_astar_experiments():
    print()
    print(
        'Solve the truck deliveries problem (small input, only distance objective, ID-A*, MSTAirDist heuristics).')

    small_delivery_problem_with_distance_cost = get_deliveries_problem('small', OptimizationObjective.Distance)

    # Ex.xxx
    # TODO: create an instance of `IDAStar` once with the `TruckDeliveriesMSTAirDistHeuristic`,
    #       solve the `small_delivery_problem_with_distance_cost` with it and print the results.
    # exit()  # TODO: remove!
    id_astar = IDAStar(TruckDeliveriesMSTAirDistHeuristic, deepening_technique='iterative',
                       max_nr_states_to_expand=30_000, max_nr_iterations=400)
    res = id_astar.solve_problem(small_delivery_problem_with_distance_cost)
    print(res)

    # Ex.xxx
    # TODO: create an instance of `IDAStar` once with the `TruckDeliveriesMSTAirDistHeuristic`,
    #       solve the `small_delivery_problem_with_distance_cost` with it and print the results.
    # exit()  # TODO: remove!
    id_astar = IDAStar(TruckDeliveriesMSTAirDistHeuristic, deepening_technique='binary_search',
                       max_nr_states_to_expand=30_000, max_nr_iterations=400)
    res = id_astar.solve_problem(small_delivery_problem_with_distance_cost)
    print(res)


def run_all_experiments():
    toy_map_problem_experiments()
    basic_deliveries_truck_problem_experiments()
    deliveries_truck_problem_with_astar_experiments()
    deliveries_truck_problem_with_weighted_astar_experiments()
    multiple_objectives_deliveries_truck_problem_experiments()
    deliveries_truck_problem_with_astar_epsilon_experiments()
    deliveries_truck_problem_anytime_astar_experiments()
    big_deliveries_truck_problem_with_non_acceptable_heuristic_and_anytime_astar_experiments()
    deliveries_truck_problem_with_id_astar_experiments()


if __name__ == '__main__':
    run_all_experiments()
