from .tests_utils import *


__all__ = ['DeliveriesTestsSuitCreator']


class DeliveriesTestsSuitCreator:

    @staticmethod
    def create_tests_suit() -> SubmissionTestsSuit:
        tests_suit = SubmissionTestsSuit()
        DeliveriesTestsSuitCreator._create_map_tests(tests_suit)
        DeliveriesTestsSuitCreator._create_relaxed_deliveries_tests(tests_suit)
        DeliveriesTestsSuitCreator._create_strict_deliveries_tests(tests_suit)
        return tests_suit

    @staticmethod
    def _create_map_tests(tests_suit: SubmissionTestsSuit):
        simple_map_problem_factory = ProblemFactory(name='MapProblem', params=(54, 549))
        tests_suit.create_test(
            problem_factory=simple_map_problem_factory,
            solver_factory=SolverFactory(name='UniformCost'),
            execution_timeout=25)
        tests_suit.create_test(
            problem_factory=simple_map_problem_factory,
            solver_factory=SolverFactory(name='AStar', heuristic_name='NullHeuristic'),
            execution_timeout=25)
        tests_suit.create_astar_tests_for_weights_in_range(
            'AirDistHeuristic', simple_map_problem_factory, (15, 10), n=5)

    @staticmethod
    def _create_relaxed_deliveries_tests(tests_suit: SubmissionTestsSuit):
        big_deliveries_problem_factory = ProblemFactory(
            name='RelaxedDeliveriesProblem',
            input_name='test1_big_delivery')

        tests_suit.create_test(
            problem_factory=big_deliveries_problem_factory,
            solver_factory=SolverFactory(name='AStar', heuristic_name='MaxAirDistHeuristic'),
            execution_timeout=20)
        tests_suit.create_astar_tests_for_weights_in_range(
            'MSTAirDistHeuristic', big_deliveries_problem_factory, (15, 10), n=5)

        num_experiments = 4
        for i in range(num_experiments):
            tests_suit.create_test(
                problem_factory=big_deliveries_problem_factory,
                solver_factory=SolverFactory(name='GreedyStochastic', heuristic_name='MSTAirDistHeuristic'),
                execution_timeout=15)

    @staticmethod
    def _create_strict_deliveries_tests(tests_suit: SubmissionTestsSuit):
        small_strict_deliveries_problem_factory = ProblemFactory(
            name='StrictDeliveriesProblem',
            input_name='test1_small_delivery',
            inner_problem_solver=SolverFactory(name='AStar', heuristic_name='AirDistHeuristic'))

        tests_suit.create_astar_tests_for_weights_in_range(
            'MSTAirDistHeuristic', small_strict_deliveries_problem_factory, (40, 15), n=5)

        tests_suit.create_test(
            problem_factory=small_strict_deliveries_problem_factory,
            solver_factory=SolverFactory(name='AStar', heuristic_name='RelaxedDeliveriesHeuristic'),
            execution_timeout=50)

