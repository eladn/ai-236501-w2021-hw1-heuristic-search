from .tests_utils import *
import types


__all__ = ['DeliveriesTestsSuitCreator']


"""
TODO:
    1. Use staff sol `cached_map_distance_finder.py` everywhere
    2. Add dedicated `cached_map_distance_finder.py` test
    3. Add time + money costs tests (+ their heuristics)
    4. Fix A*eps tests
    5. Fix AnytimeA* tests
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
        DeliveriesTestsSuitCreator._create_deliveries_truck_problem_test(tests_suit)
        DeliveriesTestsSuitCreator._create_deliveries_max_air_heuristic_test(tests_suit)
        DeliveriesTestsSuitCreator._create_deliveries_sum_air_heuristic_test(tests_suit)
        DeliveriesTestsSuitCreator._create_deliveries_mst_air_heuristic_test(tests_suit)
        DeliveriesTestsSuitCreator._create_astar_and_deliveries_truck_problem_test(tests_suit)
        DeliveriesTestsSuitCreator._create_astar_and_deliveries_truck_problem_and_deliveries_sum_air_heuristic_test(tests_suit)
        return tests_suit

    @staticmethod
    def _create_map_problem_tests(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        tests_suit.create_test(
            problem_factory=simple_map_problem_factory,
            solver_factory=SolverFactory(name='AStar', heuristic_name='AirDistHeuristic'),
            files_to_override_from_staff_solution=('deliveries/map_heuristics.py', 'framework/graph_search/astar.py'),
            execution_timeout=90)

    @staticmethod
    def _create_map_air_dist_heuristic_tests(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        tests_suit.create_test(
            problem_factory=simple_map_problem_factory,
            solver_factory=SolverFactory(name='AStar', heuristic_name='AirDistHeuristic'),
            files_to_override_from_staff_solution=('deliveries/map_problem.py', 'framework/graph_search/astar.py'),
            execution_timeout=90)

    @staticmethod
    def _create_basic_astar_test(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')
        for w in (0.5, 0.6, 0.7):
            tests_suit.create_test(
                problem_factory=simple_map_problem_factory,
                solver_factory=SolverFactory(name='AStar', heuristic_name='AirDistHeuristic', params=(w,)),
                files_to_override_from_staff_solution=('deliveries/map_problem.py', 'deliveries/map_heuristics.py'),
                execution_timeout=90)
            tests_suit.create_test(
                problem_factory=small_deliveries_problem_factory,
                solver_factory=SolverFactory(name='AStar', heuristic_name='TruckDeliveriesMSTAirDistHeuristic', params=(w,)),
                fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
                files_to_override_from_staff_solution=('deliveries/deliveries_truck_problem.py', 'deliveries/deliveries_truck_heuristics.py'),
                execution_timeout=90)
        tests_suit.create_test(
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(name='AStar', heuristic_name='TruckDeliveriesSumAirDistHeuristic', params=(0.7,)),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('deliveries/deliveries_truck_problem.py', 'deliveries/deliveries_truck_heuristics.py'),
            execution_timeout=90)

    @staticmethod
    def _create_basic_astar_epsilon_test(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        # TODO: fix it!

        tests_suit.create_test(
            problem_factory=simple_map_problem_factory,
            solver_factory=SolverFactory(name='AStarEpsilon', heuristic_name='AirDistHeuristic', ctor_kwargs={}),
            files_to_override_from_staff_solution=('deliveries/map_problem.py', 'deliveries/map_heuristics.py'),
            execution_timeout=90)
        tests_suit.create_test(
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(name='AStarEpsilon', heuristic_name='TruckDeliveriesMSTAirDistHeuristic', ctor_kwargs={}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('deliveries/deliveries_truck_problem.py', 'deliveries/deliveries_truck_heuristics.py'),
            execution_timeout=90)
        tests_suit.create_test(
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(name='AStarEpsilon', heuristic_name='TruckDeliveriesSumAirDistHeuristic', ctor_kwargs={}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('deliveries/deliveries_truck_problem.py', 'deliveries/deliveries_truck_heuristics.py'),
            execution_timeout=90)


    @staticmethod
    def _create_basic_anytime_astar_test(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        # TODO: fix it!

        tests_suit.create_test(
            problem_factory=simple_map_problem_factory,
            solver_factory=SolverFactory(name='AnytimeAStar', heuristic_name='AirDistHeuristic', ctor_kwargs={}),
            files_to_override_from_staff_solution=('deliveries/map_problem.py', 'deliveries/map_heuristics.py'),
            execution_timeout=90)
        tests_suit.create_test(
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(name='AnytimeAStar', heuristic_name='TruckDeliveriesMSTAirDistHeuristic', ctor_kwargs={}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('deliveries/deliveries_truck_problem.py', 'deliveries/deliveries_truck_heuristics.py'),
            execution_timeout=90)
        tests_suit.create_test(
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(name='AnytimeAStar', heuristic_name='TruckDeliveriesSumAirDistHeuristic', ctor_kwargs={}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('deliveries/deliveries_truck_problem.py', 'deliveries/deliveries_truck_heuristics.py'),
            execution_timeout=90)

    @staticmethod
    def _create_deliveries_truck_problem_test(tests_suit: SubmissionTestsSuit):
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic_name='TruckDeliveriesSumAirDistHeuristic'),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_heuristics.py'),
            execution_timeout=90)
        tests_suit.create_test(
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic_name='TruckDeliveriesSumAirDistHeuristic', ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_heuristics.py'),
            execution_timeout=90)

    @staticmethod
    def _create_deliveries_max_air_heuristic_test(tests_suit: SubmissionTestsSuit):
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic_name='TruckDeliveriesMaxAirDistHeuristic'),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py'),
            execution_timeout=90)
        tests_suit.create_test(
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic_name='TruckDeliveriesMaxAirDistHeuristic', ctor_kwargs={'heuristic_weight': 0.85}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py'),
            execution_timeout=90)

    @staticmethod
    def _create_deliveries_sum_air_heuristic_test(tests_suit: SubmissionTestsSuit):
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic_name='TruckDeliveriesSumAirDistHeuristic'),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py'),
            execution_timeout=90)
        tests_suit.create_test(
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic_name='TruckDeliveriesSumAirDistHeuristic', ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py'),
            execution_timeout=90)

    @staticmethod
    def _create_deliveries_mst_air_heuristic_test(tests_suit: SubmissionTestsSuit):
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic_name='TruckDeliveriesMSTAirDistHeuristic'),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py'),
            execution_timeout=90)
        tests_suit.create_test(
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic_name='TruckDeliveriesMSTAirDistHeuristic', ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('framework/graph_search/astar.py', 'deliveries/deliveries_truck_problem.py'),
            execution_timeout=90)

    @staticmethod
    def _create_astar_and_deliveries_truck_problem_test(tests_suit: SubmissionTestsSuit):
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic_name='TruckDeliveriesSumAirDistHeuristic'),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('deliveries/deliveries_truck_heuristics.py',),
            execution_timeout=90)
        tests_suit.create_test(
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic_name='TruckDeliveriesSumAirDistHeuristic', ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            files_to_override_from_staff_solution=('deliveries/deliveries_truck_heuristics.py',),
            execution_timeout=90)

    @staticmethod
    def _create_astar_and_deliveries_truck_problem_and_deliveries_sum_air_heuristic_test(tests_suit: SubmissionTestsSuit):
        small_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_small')
        medium_deliveries_problem_factory = ProblemFactory(name='DeliveriesTruckProblem', input_name='test_deliveries_medium')

        tests_suit.create_test(
            problem_factory=small_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic_name='TruckDeliveriesSumAirDistHeuristic'),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            execution_timeout=90)
        tests_suit.create_test(
            problem_factory=medium_deliveries_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic_name='TruckDeliveriesSumAirDistHeuristic', ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_delivery_problem_method__all_junctions_in_remaining_truck_path,
            execution_timeout=90)
