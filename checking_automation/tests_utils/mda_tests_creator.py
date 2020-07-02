from .tests_utils import *
import types


__all__ = ['MDATestsSuitCreator']


def fix_mda_problem_method__all_junctions_in_remaining_ambulance_path(mda_problem, solver):
    mda_problem.__get_reported_apartments_waiting_to_visit = \
        mda_problem.get_reported_apartments_waiting_to_visit

    def new__get_reported_apartments_waiting_to_visit(_mda_problem, state):
        all_apartments = list(_mda_problem.__get_reported_apartments_waiting_to_visit(state))
        all_apartments.sort(key=lambda apartment: apartment.location.index)
        return set(list(all_apartments))

    mda_problem.get_reported_apartments_waiting_to_visit = types.MethodType(
        new__get_reported_apartments_waiting_to_visit, mda_problem)


class MDATestsSuitCreator:

    @staticmethod
    def create_tests_suit() -> SubmissionTestsSuit:
        tests_suit = SubmissionTestsSuit()
        MDATestsSuitCreator._create_map_problem_tests(tests_suit)
        MDATestsSuitCreator._create_map_air_dist_heuristic_tests(tests_suit)
        MDATestsSuitCreator._create_basic_astar_test(tests_suit)
        MDATestsSuitCreator._create_basic_astar_epsilon_test(tests_suit)
        MDATestsSuitCreator._create_basic_anytime_astar_test(tests_suit)
        MDATestsSuitCreator._create_cached_map_distance_finder_test(tests_suit)
        MDATestsSuitCreator._create_mda_problem_test(tests_suit)
        MDATestsSuitCreator._create_mda_problem_with_tests_travel_distance_objective_test(tests_suit)
        MDATestsSuitCreator._create_mda_max_air_heuristic_test(tests_suit)
        MDATestsSuitCreator._create_mda_sum_air_heuristic_test(tests_suit)
        MDATestsSuitCreator._create_mda_mst_air_heuristic_test(tests_suit)
        MDATestsSuitCreator._create_astar_and_mda_problem_test(tests_suit)
        MDATestsSuitCreator._create_astar_and_mda_problem_and_mda_sum_air_heuristic_overall_test(tests_suit)
        return tests_suit

    @staticmethod
    def _create_map_problem_tests(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        tests_suit.create_test(
            name='map_problem',
            problem_factory=simple_map_problem_factory,
            solver_factory=SolverFactory(name='AStar', heuristic=HeuristicFactory('AirDistHeuristic')),
            files_to_override_from_staff_solution=('problems/map_heuristics.py', 'framework/graph_search/astar.py'),
            execution_timeout=90)

    @staticmethod
    def _create_map_air_dist_heuristic_tests(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        tests_suit.create_test(
            name='map_air_dist_heuristic',
            problem_factory=simple_map_problem_factory,
            solver_factory=SolverFactory(name='AStar', heuristic=HeuristicFactory('AirDistHeuristic')),
            files_to_override_from_staff_solution=('problems/map_problem.py', 'framework/graph_search/astar.py'),
            execution_timeout=90)

    @staticmethod
    def _create_basic_astar_test(tests_suit: SubmissionTestsSuit):
        simple_map_problem1_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        simple_map_problem2_factory = ProblemFactory(name='MapProblem', params=(123, 735))
        small_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_small_input')
        medium_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_medium_input')
        for w in (0.5, 0.6, 0.7):
            for map_problem in (simple_map_problem1_factory, simple_map_problem2_factory):
                tests_suit.create_test(
                    name='astar',
                    problem_factory=map_problem,
                    solver_factory=SolverFactory(
                        name='AStar', heuristic=HeuristicFactory('AirDistHeuristic'), params=(w,)),
                    files_to_override_from_staff_solution=('problems/map_problem.py', 'problems/map_heuristics.py'),
                    execution_timeout=90)
            tests_suit.create_test(
                name='astar',
                problem_factory=small_mda_problem_factory,
                solver_factory=SolverFactory(
                    name='AStar', heuristic=HeuristicFactory('MDASumAirDistHeuristicForTests'),
                    params=(w,)),
                fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
                files_to_override_from_staff_solution=(
                    'problems/mda_problem.py', 'problems/mda_heuristics.py', 'problems/__init__.py',
                    'problems/cached_map_distance_finder.py'),
                execution_timeout=90)
        tests_suit.create_test(
            name='astar',
            problem_factory=medium_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDASumAirDistHeuristicForTests'), params=(0.7,)),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'problems/mda_problem.py', 'problems/mda_heuristics.py', 'problems/__init__.py',
                'problems/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_basic_astar_epsilon_test(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        small_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_small_input')
        # medium_mda_problem_factory = ProblemFactory(
        #     name='MDAProblem', input_name='test_mda_medium_input')

        def within_focal_map_h_priority_function(node, problem, solver):
            # expanding_priority == solver.heuristic_weight * h + (1-solver.heuristic_weight) * g_cost
            # ==>  h == (expanding_priority + (solver.heuristic_weight-1) * g_cost) / solver.heuristic_weight
            return (node.expanding_priority + (solver.heuristic_weight - 1) * node.g_cost) / solver.heuristic_weight

        def within_focal_mda_h_sum_priority_function(node, problem, solver):
            if not hasattr(solver, '__focal_heuristic'):
                heuristic_ctor = HeuristicFactory('MDASumAirDistHeuristicForTests').get_heuristic_ctor()
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
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'problems/map_problem.py', 'problems/map_heuristics.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='astar_epsilon',
            problem_factory=small_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStarEpsilon', heuristic=HeuristicFactory('MDAMSTAirDistHeuristic'), ctor_kwargs={
                    'max_nr_states_to_expand': 2_000, 'max_focal_size': 40, 'focal_epsilon': 0.03,
                    'within_focal_priority_function': within_focal_mda_h_sum_priority_function}),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'problems/mda_problem.py', 'problems/mda_heuristics.py',
                'problems/__init__.py', 'problems/cached_map_distance_finder.py'),
            execution_timeout=200)
        # tests_suit.create_test(
        #     name='astar_epsilon',
        #     problem_factory=medium_mda_problem_factory,
        #     solver_factory=SolverFactory(
        #         name='AStarEpsilon', heuristic=HeuristicFactory('MDAMSTAirDistHeuristic'), ctor_kwargs={
        #             'max_nr_states_to_expand': 31_000, 'max_focal_size': 30, 'focal_epsilon': 0.04,
        #             'within_focal_priority_function': within_focal_mda_h_sum_priority_function}),
        #     fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
        #     files_to_override_from_staff_solution=(
        #         'framework/graph_search/astar.py', 'problems/mda_problem.py', 'problems/mda_heuristics.py',
        #         'problems/__init__.py', 'problems/cached_map_distance_finder.py'),
        #     execution_timeout=400)

    @staticmethod
    def _create_basic_anytime_astar_test(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(63, 947))
        small_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_small_input')
        medium_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_medium_input')

        tests_suit.create_test(
            name='anytime_astar',
            problem_factory=simple_map_problem_factory,
            solver_factory=SolverFactory(
                name='AnytimeAStar', heuristic=HeuristicFactory('AirDistHeuristic'),
                ctor_kwargs={'max_nr_states_to_expand_per_iteration': 1500,
                             'initial_high_heuristic_weight_bound': 0.8}),
            files_to_override_from_staff_solution=(
                'problems/map_problem.py', 'problems/map_heuristics.py', 'framework/graph_search/astar.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='anytime_astar',
            problem_factory=small_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AnytimeAStar', heuristic=HeuristicFactory('MDASumAirDistHeuristicForTests'),
                ctor_kwargs={'max_nr_states_to_expand_per_iteration': 80,
                             'initial_high_heuristic_weight_bound': 0.8}),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'problems/mda_problem.py', 'problems/mda_heuristics.py',
                'problems/__init__.py', 'problems/cached_map_distance_finder.py',
                'framework/graph_search/astar.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='anytime_astar',
            problem_factory=medium_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AnytimeAStar', heuristic=HeuristicFactory('MDASumAirDistHeuristicForTests'),
                ctor_kwargs={'max_nr_states_to_expand_per_iteration': 200,
                             'initial_high_heuristic_weight_bound': 0.8}),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'problems/mda_problem.py', 'problems/mda_heuristics.py',
                'problems/__init__.py', 'problems/cached_map_distance_finder.py',
                'framework/graph_search/astar.py'),
            execution_timeout=90)

    @staticmethod
    def _create_mda_problem_test(tests_suit: SubmissionTestsSuit):
        small_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_small_input')
        medium_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_medium_input')

        tests_suit.create_test(
            name='mda_problem',
            problem_factory=small_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDASumAirDistHeuristicForTests')),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'problems/mda_heuristics.py', 'problems/__init__.py',
                'problems/cached_map_distance_finder.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='mda_problem',
            problem_factory=medium_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDASumAirDistHeuristicForTests'),
                ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'problems/mda_heuristics.py', 'problems/__init__.py',
                'problems/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_mda_problem_with_tests_travel_distance_objective_test(tests_suit: SubmissionTestsSuit):
        money_small_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_small_input',
            kwargs_builder={
                'optimization_objective': lambda: get_module_by_name('MDAOptimizationObjective').TestsTravelDistance})

        tests_suit.create_test(
            name='tests_travel_distance_objective_mda_problem',
            problem_factory=money_small_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDATestsTravelDistToNearestLabHeuristic')),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
            'framework/graph_search/astar.py', 'problems/mda_heuristics.py', 'problems/__init__.py',
            'problems/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_cached_map_distance_finder_test(tests_suit: SubmissionTestsSuit):
        small_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_small_input')
        medium_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_medium_input')

        tests_suit.create_test(
            name='cached_map_distance_finder',
            problem_factory=small_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDASumAirDistHeuristicForTests')),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'problems/mda_problem.py',
                'problems/mda_heuristics.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='cached_map_distance_finder',
            problem_factory=medium_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDASumAirDistHeuristicForTests'),
                ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'problems/mda_problem.py',
                'problems/mda_heuristics.py'),
            execution_timeout=90)

    @staticmethod
    def _create_mda_max_air_heuristic_test(tests_suit: SubmissionTestsSuit):
        small_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_small_input')
        medium_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_medium_input')

        tests_suit.create_test(
            name='mda_max_air_heuristic',
            problem_factory=small_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDAMaxAirDistHeuristic')),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'problems/mda_problem.py',
                'problems/cached_map_distance_finder.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='mda_max_air_heuristic',
            problem_factory=medium_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDAMaxAirDistHeuristic'),
                ctor_kwargs={'heuristic_weight': 0.85}),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'problems/mda_problem.py',
                'problems/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_mda_sum_air_heuristic_test(tests_suit: SubmissionTestsSuit):
        small_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_small_input')
        medium_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_medium_input')

        tests_suit.create_test(
            name='mda_sum_air_heuristic',
            problem_factory=small_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDASumAirDistHeuristic')),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'problems/mda_problem.py',
                'problems/cached_map_distance_finder.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='mda_sum_air_heuristic',
            problem_factory=medium_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDASumAirDistHeuristic'),
                ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'problems/mda_problem.py',
                'problems/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_mda_mst_air_heuristic_test(tests_suit: SubmissionTestsSuit):
        small_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_small_input')
        medium_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_medium_input')

        tests_suit.create_test(
            name='mda_mst_air_heuristic',
            problem_factory=small_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDAMSTAirDistHeuristic')),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'problems/mda_problem.py',
                'problems/cached_map_distance_finder.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='mda_mst_air_heuristic',
            problem_factory=medium_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDAMSTAirDistHeuristic'),
                ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'framework/graph_search/astar.py', 'problems/mda_problem.py',
                'problems/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_astar_and_mda_problem_test(tests_suit: SubmissionTestsSuit):
        small_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_small_input')
        medium_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_medium_input')

        tests_suit.create_test(
            name='astar_and_mda_problem',
            problem_factory=small_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDASumAirDistHeuristicForTests')),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'problems/mda_heuristics.py', 'problems/__init__.py', 'problems/cached_map_distance_finder.py'),
            execution_timeout=90)
        tests_suit.create_test(
            name='astar_and_mda_problem',
            problem_factory=medium_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDASumAirDistHeuristicForTests'),
                ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            files_to_override_from_staff_solution=(
                'problems/mda_heuristics.py', 'problems/__init__.py', 'problems/cached_map_distance_finder.py'),
            execution_timeout=90)

    @staticmethod
    def _create_astar_and_mda_problem_and_mda_sum_air_heuristic_overall_test(
            tests_suit: SubmissionTestsSuit):
        small_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_small_input')
        medium_mda_problem_factory = ProblemFactory(
            name='MDAProblem', input_name='test_mda_medium_input')

        tests_suit.create_test(
            name='overall',
            problem_factory=small_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDASumAirDistHeuristic')),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            execution_timeout=90)
        tests_suit.create_test(
            name='overall',
            problem_factory=medium_mda_problem_factory,
            solver_factory=SolverFactory(
                name='AStar', heuristic=HeuristicFactory('MDASumAirDistHeuristic'),
                ctor_kwargs={'heuristic_weight': 0.7}),
            fn_to_execute_before_solving=fix_mda_problem_method__all_junctions_in_remaining_ambulance_path,
            execution_timeout=90)
