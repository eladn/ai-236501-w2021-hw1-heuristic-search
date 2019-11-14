from framework import *
from deliveries import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union

# Load the streets map
streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path("tlv.csv"))

# Make sure that the whole execution is deterministic.
# This is important, because we expect to get the exact same results
# in each execution.
Consts.set_seed()


# --------------------------------------------------------------------
# ------------------------ StreetsMap Problem ------------------------
# --------------------------------------------------------------------

def plot_distance_and_expanded_wrt_weight_figure(
        weights: Union[np.ndarray, List[float]],
        total_distance: Union[np.ndarray, List[float]],
        total_expanded: Union[np.ndarray, List[int]]):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    """
    assert len(weights) == len(total_distance) == len(total_expanded)

    fig, ax1 = plt.subplots()

    # TODO: Plot the total distances with ax1. Use `ax1.plot(...)`.
    # TODO: Make this curve colored blue with solid line style.
    # See documentation here:
    # https://matplotlib.org/2.0.0/api/_as_gen/matplotlib.axes.Axes.plot.html
    # You can also search google for additional examples.
    # raise NotImplemented()
    ax1.plot(weights, total_distance, 'b-')

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('distance traveled', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()

    # TODO: Plot the total expanded with ax2. Use `ax2.plot(...)`.
    # TODO: ax2: Make the y-axis label, ticks and tick labels match the line color.
    # TODO: Make this curve colored red with solid line style.
    # raise NotImplemented()
    ax2.plot(weights, total_expanded, 'r-')
    ax2.set_ylabel('states expanded', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem):
    # TODO:
    #  1. Create an array of 20 numbers equally spread in [0.5, 1]
    #     (including the edges). You can use `np.linspace()` for that.
    #  2. For each weight in that array run the A* algorithm, with the
    #     given `heuristic_type` over the given problem. For each such run,
    #     if a solution has been found (res.is_solution_found), store the
    #     cost of the solution (res.solution_g_cost), the number of
    #     expanded states (res.nr_expanded_states), and the weight that
    #     has been used in this iteration. Store these in 3 lists (array
    #     for the costs, array for the #expanded and array for the weights.
    #     These arrays should be of the same size when this operation ends.
    #  Call the function `plot_distance_and_expanded_by_weight_figure()`
    #   with these 3 generated arrays.
    # raise NotImplemented()  # TODO: remove!
    total_cost = []
    total_nr_expanded = []
    weights = np.linspace(0.5, 1, 20)
    weights_with_found_solutions = []
    for w in weights:
        astar = AStar(heuristic_type, w, max_nr_states_to_expand=30_000)
        res = astar.solve_problem(problem)
        print(res)
        if res.is_solution_found:
            total_cost.append(res.solution_g_cost)
            total_nr_expanded.append(res.nr_expanded_states)
            weights_with_found_solutions.append(w)
    plot_distance_and_expanded_wrt_weight_figure(weights_with_found_solutions, total_cost, total_nr_expanded)


def map_problem_experiments():
    print()
    print('Solve the map problem.')

    # Ex.8
    toy_map_problem = MapProblem(streets_map, 54, 549)
    uc = UniformCost()
    res = uc.solve_problem(toy_map_problem)
    print(res)

    # Ex.10
    # TODO: create an instance of `AStar` with the `NullHeuristic`,
    #       solve the same `toy_map_problem` with it and print the results (as before).
    # Notice: AStar constructor receives the heuristic *type* (ex: `MyHeuristicClass`),
    #         and NOT an instance of the heuristic (eg: not `MyHeuristicClass()`).
    # exit()  # TODO: remove!
    astar = AStar(NullHeuristic)
    res = astar.solve_problem(toy_map_problem)
    print(res)

    # Ex.11
    # TODO: create an instance of `AStar` with the `AirDistHeuristic`,
    #       solve the same `toy_map_problem` with it and print the results (as before).
    # exit()  # TODO: remove!
    astar = AStar(AirDistHeuristic)
    res = astar.solve_problem(toy_map_problem)
    print(res)

    # Ex.12
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

def deliveries_truck_problem_experiments():
    print()
    print('Solve the truck deliveries problem.')

    big_delivery = DeliveriesTruckProblemInput.load_from_file('big_delivery.in', streets_map)
    big_deliveries_prob = DeliveriesTruckProblem(problem_input=big_delivery, streets_map=streets_map)

    # Ex.16
    # TODO: create an instance of `AStar` with the `TruckDeliveriesMaxAirDistHeuristic`,
    #       solve the `big_deliveries_prob` with it and print the results (as before).
    # exit()  # TODO: remove!
    astar = AStar(TruckDeliveriesMaxAirDistHeuristic)
    res = astar.solve_problem(big_deliveries_prob)
    print(res)

    # Ex.17
    # TODO: create an instance of `AStar` with the `TruckDeliveriesMSTAirDistHeuristic`,
    #       solve the `big_deliveries_prob` with it and print the results (as before).
    # exit()  # TODO: remove!
    astar = AStar(TruckDeliveriesMSTAirDistHeuristic)
    res = astar.solve_problem(big_deliveries_prob)
    print(res)

    # Ex.18
    # TODO: Call here the function `run_astar_for_weights_in_range()`
    #       with `TruckDeliveriesMSTAirDistHeuristic` and `big_deliveries_prob`.
    # exit()  # TODO: remove!
    run_astar_for_weights_in_range(TruckDeliveriesMSTAirDistHeuristic, big_deliveries_prob)

    # Ex.24
    # TODO:
    # 1. Run the stochastic greedy algorithm for 100 times.
    #    For each run, store the cost of the found solution.
    #    Store these costs in a list.
    # 2. The "Anytime Greedy Stochastic Algorithm" runs the greedy
    #    greedy stochastic for N times, and after each iteration
    #    stores the best solution found so far. It means that after
    #    iteration #i, the cost of the solution found by the anytime
    #    algorithm is the MINIMUM among the costs of the solutions
    #    found in iterations {1,...,i}. Calculate the costs of the
    #    anytime algorithm wrt the #iteration and store them in a list.
    # 3. Calculate and store the cost of the solution received by
    #    the A* algorithm (with w=0.5).
    # 4. Calculate and store the cost of the solution received by
    #    the deterministic greedy algorithm (A* with w=1).
    # 5. Plot a figure with the costs (y-axis) wrt the #iteration
    #    (x-axis). Of course that the costs of A*, and deterministic
    #    greedy are not dependent with the iteration number, so
    #    these two should be represented by horizontal lines.
    # exit()  # TODO: remove!
    astar = AStar(MSTAirDistHeuristic)
    astar_res = astar.solve_problem(big_deliveries_prob)
    astar_cost = astar_res.final_search_node.cost
    astar_developed = astar_res.nr_expanded_states
    stochastic_costs = []
    stochastic_developed = []
    num_experiments = 100
    for i in range(num_experiments):
        greedy = GreedyStochastic(MSTAirDistHeuristic)
        cur_greedy_stochastic_res = greedy.solve_problem(big_deliveries_prob)
        stochastic_costs.append(cur_greedy_stochastic_res.final_search_node.cost)
        stochastic_developed.append(cur_greedy_stochastic_res.nr_expanded_states)
    anytime_stochastic_costs = [np.min(stochastic_costs[:i+1]) for i in range(num_experiments)]
    greey_solver = AStar(MSTAirDistHeuristic, heuristic_weight=1)
    greedy_res = greey_solver.solve_problem(big_deliveries_prob)
    greedy_cost = greedy_res.final_search_node.cost
    greedy_developed = greedy_res.nr_expanded_states

    plt.figure()
    plt.plot(stochastic_costs, label='greedy_stochastic', alpha=0.5, color='grey')
    plt.plot([0, num_experiments-1], [greedy_cost, greedy_cost], label='greedy')
    plt.plot(anytime_stochastic_costs, label='anytime_stochastic')
    plt.plot([0, num_experiments-1], [astar_cost, astar_cost], label='A*')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(stochastic_developed, label='greedy_stochastic', alpha=0.5, color='grey')
    plt.plot([0, num_experiments - 1], [greedy_developed, greedy_developed], label='greedy')
    plt.plot([0, num_experiments - 1], [astar_developed, astar_developed], label='A*')
    plt.legend()
    plt.show()


def main():
    map_problem_experiments()
    deliveries_truck_problem_experiments()


if __name__ == '__main__':
    main()
