from .tests_utils import *
import types


__all__ = ['DeliveriesTestsSuitCreator']


"""
Done:
    [V] Use staff sol `deliveries/cached_map_distance_finder.py` everywhere
    [V] Add dedicated `deliveries/cached_map_distance_finder.py` test
    [V] Add time + money costs tests (+ their lower bounds calculation for heuristics)
    [V] Fix A*eps tests
    [V] Fix AnytimeA* tests
    [V] In A*eps + AnytimeA* tests - use the stuff-sol A*
    [V] Sum deliveries heuristic which is intended for the tests (sorts junctions by secondary sorting)
    [V] add descriptive names to tests
"""


def fix_delivery_problem_method__all_junctions_in_remaining_truck_path(delivery_problem, solver):
    delivery_problem.__get_all_junctions_in_remaining_truck_path = delivery_problem.get_all_junctions_in_remaining_truck_path

    def new__get_all_junctions_in_remaining_truck_path(deliveries_problem, state):
        all_junctions = list(deliveries_problem.__get_all_junctions_in_remaining_truck_path(state))
        all_junctions.sort(key=lambda junction: junction.index)
        return set(list(all_junctions))

    delivery_problem.get_all_junctions_in_remaining_truck_path = types.MethodType(
        new__get_all_junctions_in_remaining_truck_path, delivery_problem)


class DeliveriesTestsSuitCreator:

    @staticmethod
    def create_tests_suit() -> SubmissionTestsSuit:
        tests_suit = SubmissionTestsSuit()
        DeliveriesTestsSuitCreator._create_map_problem_tests(tests_suit)
        DeliveriesTestsSuitCreator._create_map_air_dist_heuristic_tests(tests_suit)
        DeliveriesTestsSuitCreator._create_basic_astar_test(tests_suit)
        DeliveriesTestsSuitCreator._create_basic_astar_epsilon_test(tests_suit)
        DeliveriesTestsSuitCreator._create_basic_anytime_astar_test(tests_suit)
        DeliveriesTestsSuitCreator._create_cached_map_distance_finder_test(tests_suit)
        DeliveriesTestsSuitCreator._create_deliveries_truck_problem_test(tests_suit)
        DeliveriesTestsSuitCreator._create_deliveries_truck_problem_with_money_and_time_objectives_test(tests_suit)
        DeliveriesTestsSuitCreator._create_deliveries_max_air_heuristic_test(tests_suit)
        DeliveriesTestsSuitCreator._create_deliveries_sum_air_heuristic_test(tests_suit)
        DeliveriesTestsSuitCreator._create_deliveries_mst_air_heuristic_test(tests_suit)
        DeliveriesTestsSuitCreator._create_astar_and_deliveries_truck_problem_test(tests_suit)
        DeliveriesTestsSuitCreator._create_astar_and_deliveries_truck_problem_and_deliveries_sum_air_heuristic_overall_test(tests_suit)
        return tests_suit

    @staticmethod
    def _create_map_problem_tests(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        tests_suit.create_test(
            name='map_problem',
            problem_factory=simple_map_problem_factory,
            solver_factory=SolverFactory(name='AStar', heuristic=HeuristicFactory('AirDistHeuristic')),
            files_to_override_from_staff_solution=('deliveries/map_heuristics.py', 'framework/graph_search/astar.py'),
            execution_timeout=90)

    @staticmethod
    def _create_map_air_dist_heuristic_tests(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        tests_suit.create_test(
            name='map_air_dist_heuristic',
            problem_factory=simple_map_problem_factory,
            solver_factory=SolverFactory(name='AStar', heuristic=HeuristicFactory('AirDistHeuristic')),
            files_to_override_from_staff_solution=('deliveries/map_problem.py', 'framework/graph_search/astar.py'),
            execution_timeout=90)

    @staticmethod
    def _create_basic_astar_test(tests_suit: SubmissionTestsSuit):
        simple_map_problem1_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        simple_map_problem2_factory = ProblemFactory(name='MapProblem', params=(123, 735))
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')
        for w in (0.5, 0.6, 0.7):
            for map_problem in (simple_map_problem1_factory, simple_map_problem2_factory):
                tests_suit.create_test(
                    name='astar',
                    problem_factory=map_problem,
                    solver_factory=SolverFactory(name='AStar', heuristic=HeuristicFactory('AirDistHeuristic'), params=(w,)),
                    files_to_override_from_staff_solution=('deliveries/map_problem.py', 'deliveries/map_heuristics.py'),
                    execution_timeout=90)
            tests_suit.create_test(
                name='astar',
                problem_factory=small_deliveries_problem_factory,
                solver_factory=SolverFactory(name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristicForTests'), params=(w,)),
                fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
                files_to_override_from_staff_solution=('deliveries/deliveries_truck_problem.py', 'deliveries/deliveries_truck_heuristics.py', 'deliveries/cached_map_distance_finder.py'),
                execution_timeout=90)
        tests_suit.create_test(
            name='astar',
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristicForTests'), params=(0.7,)),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('deliveries/deliveries_truck_problem.py', 'deliveries/deliveries_truck_heuristics.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_basic_astar_epsilon_test(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        def within_focal_map_h_priority_function(node, problem, solver):
            # expanding_priority == solver.heuristic_weight * h + (1-solver.heuristic_weight) * g_cost
            # ==>  h == (expanding_priority + (solver.heuristic_weight-1) * g_cost) / solver.heuristic_weight
            return (node.expanding_priority + (solver.heuristic_weight - 1) * node.g_cost) / solver.heuristic_weight

        def within_focal_deliveries_h_sum_priority_function(node, problem, solver):
            if not hasattr(solver, '__focal_heuristic'):
                heuristic_ctor = HeuristicFactory('TruckDeliveriesSumAirDistHeuristicForTests').get_heuristic_ctor()
                setattr(solver, '__focal_heuristic', heuristic_ctor(problem=problem))
            focal_heuristic = getattr(solver, '__focal_heuristic')
            return focal_heuristic.estimate(node.state)

        tests_suit.create_test(
            name='astar_epsilon',
            problem_factory=simple_map_problem_factory,
            solver_factory=SolverFactory(
                name='AStarEpsilon', heuristic=HeuristicFactory('AirDistHeuristic'), ctor_kwargs={
                    'max_focal_size': 40, 'focal_epsilon': 0.1,
                    'within_focal_priority_function': within_focal_map_h_priority_function}),
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/map_problem.py', 'deliveries/map_heuristics.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='astar_epsilon',
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStarEpsilon', heuristic=HeuristicFactory('TruckDeliveriesMSTAirDistHeuristic'), ctor_kwargs={
                    'max_focal_size': 30, 'focal_epsilon': 0.04,
                    'within_focal_priority_function': within_focal_deliveries_h_sum_priority_function}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py', 'deliveries/deliveries_truck_heuristics.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=200)

    @staticmethod
    def _create_basic_anytime_astar_test(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            name='anytime_astar',
            problem_factory=simple_map_problem_factory,
            solver_factory=SolverFactory(
                name='AnytimeAStar', heuristic=HeuristicFactory('AirDistHeuristic'),
                ctor_kwargs={'max_nr_states_to_expand_per_iteration': 50}),
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/map_problem.py', 'deliveries/map_heuristics.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='anytime_astar',
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AnytimeAStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristicForTests'),
                ctor_kwargs={'max_nr_states_to_expand_per_iteration': 50}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py', 'deliveries/deliveries_truck_heuristics.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='anytime_astar',
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AnytimeAStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristicForTests'),
                ctor_kwargs={'max_nr_states_to_expand_per_iteration': 50}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py', 'deliveries/deliveries_truck_heuristics.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_deliveries_truck_problem_test(tests_suit: SubmissionTestsSuit):
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            name='deliveries_truck_problem',
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristicForTests')),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_heuristics.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='deliveries_truck_problem',
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristicForTests'), ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_heuristics.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_deliveries_truck_problem_with_money_and_time_objectives_test(tests_suit: SubmissionTestsSuit):
        money_small_deliveries_problem_factory = ProblemFactory(
            name='DeliveriesTruckProblem', input_name='test_deliveries_small',
            kwargs_builder={'optimization_objective': lambda: get_module_by_name('OptimizationObjective').Money})
        time_small_deliveries_problem_factory = ProblemFactory(
            name='DeliveriesTruckProblem', input_name='test_deliveries_small',
            kwargs_builder={'optimization_objective': lambda: get_module_by_name('OptimizationObjective').Time})

        tests_suit.create_test(
            name='money_objectives_deliveries_truck_problem',
            problem_factory=money_small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristicForTests')),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=(
            'framework/graph_search/astar.py', 'deliveries/deliveries_truck_heuristics.py',
            'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='time_objectives_deliveries_truck_problem',
            problem_factory=time_small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristicForTests')),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=(
            'framework/graph_search/astar.py', 'deliveries/deliveries_truck_heuristics.py',
            'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_cached_map_distance_finder_test(tests_suit: SubmissionTestsSuit):
        small_deliveries_problem_factory = ProblemFactory(
            name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(
            name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            name='cached_map_distance_finder',
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristicForTests')),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py',
                'deliveries/deliveries_truck_heuristics.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='cached_map_distance_finder',
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristicForTests'),
                ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py',
                'deliveries/deliveries_truck_heuristics.py'),
            execution_timeout=90)

    @staticmethod
    def _create_deliveries_max_air_heuristic_test(tests_suit: SubmissionTestsSuit):
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            name='deliveries_max_air_heuristic',
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesMaxAirDistHeuristic')),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='deliveries_max_air_heuristic',
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesMaxAirDistHeuristic'), ctor_kwargs={'heuristic_weight': 0.85}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_deliveries_sum_air_heuristic_test(tests_suit: SubmissionTestsSuit):
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            name='deliveries_sum_air_heuristic',
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristic')),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='deliveries_sum_air_heuristic',
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristic'), ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_deliveries_mst_air_heuristic_test(tests_suit: SubmissionTestsSuit):
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            name='deliveries_mst_air_heuristic',
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesMSTAirDistHeuristic')),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='deliveries_mst_air_heuristic',
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesMSTAirDistHeuristic'), ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_astar_and_deliveries_truck_problem_test(tests_suit: SubmissionTestsSuit):
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            name='astar_and_deliveries_truck_problem',
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristicForTests')),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('deliveries/deliveries_truck_heuristics.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='astar_and_deliveries_truck_problem',
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristicForTests'), ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('deliveries/deliveries_truck_heuristics.py', 'deliveries/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_astar_and_deliveries_truck_problem_and_deliveries_sum_air_heuristic_overall_test(tests_suit: SubmissionTestsSuit):
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            name='overall',
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristic')),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            execution_timeout=90)
        tests_suit.create_test(
            name='overall',
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('TruckDeliveriesSumAirDistHeuristic'), ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            execution_timeout=90)
