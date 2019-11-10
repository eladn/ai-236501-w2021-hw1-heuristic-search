
__all__ = [
    'PYTHON_INTERPRETER_FOR_SUBMISSIONS_TESTS',
    'SUBMISSIONS_PATH',
    'TESTS_ENVIRONMENTS_PATH',
    'TESTS_LOGS_PATH',
    'CLEAN_SUPPLIED_CODE_ENV_PATH',
    'STAFF_SOLUTION_CODE_PATH',
    'STAFF_SOLUTION_DUMMY_ID',
    'NUMPY_RANDOM_SEED_BASE',
    'NR_ATTEMPTS_PER_TEST',
    'TEST_TIME_OVERHEAD_EST_IN_SECONDS',
    'NR_PROCESSES',
    'CHECK_AUTOMATION_CODE_PATH',
    'NONVITAL_REQUIRED_SUBMISSION_CODE_FILES',
    'FILES_ASKED_NOT_TO_SUBMIT',
    'VITAL_REQUIRED_SUBMISSION_CODE_FILES',
    'FILES_TO_COPY_FROM_CLEAN_SUPPLIED_CODE',
    'TEST_SCRIPT_FILES',
    'TEST_SCRIPT_FILENAME'
]

PYTHON_INTERPRETER_FOR_SUBMISSIONS_TESTS = '/anaconda3/envs/py37/bin/python'
SUBMISSIONS_PATH = '/Users/eladn/Documents/ai-w2019-hw1-submissions/submissions/'
TESTS_ENVIRONMENTS_PATH = '/Users/eladn/Documents/ai-w2019-hw1-submissions/tests/'
TESTS_LOGS_PATH = '/Users/eladn/Documents/ai-w2019-hw1-submissions/tests-logs-v6/'
CLEAN_SUPPLIED_CODE_ENV_PATH = '../../supplied_code/'
STAFF_SOLUTION_CODE_PATH = '../staff_solution/'
CHECK_AUTOMATION_CODE_PATH = './'
STAFF_SOLUTION_DUMMY_ID = 1111118
NUMPY_RANDOM_SEED_BASE = 983656
NR_ATTEMPTS_PER_TEST = 2
TEST_TIME_OVERHEAD_EST_IN_SECONDS = 4  # for creating a process and load the roads
NR_PROCESSES = 2

NONVITAL_REQUIRED_SUBMISSION_CODE_FILES = [
    'main.py',
    'experiments/temperature.py'
]

FILES_ASKED_NOT_TO_SUBMIT = [
    'framework/db/tlv.csv'
]

VITAL_REQUIRED_SUBMISSION_CODE_FILES = [
    'deliveries/deliveries_heuristics.py',
    'deliveries/relaxed_deliveries_problem.py',
    'deliveries/strict_deliveries_problem.py',
    'deliveries/map_heuristics.py',
    'deliveries/map_problem.py',
    'framework/graph_search/astar.py',
    'framework/graph_search/greedy_stochastic.py'
]

FILES_TO_COPY_FROM_CLEAN_SUPPLIED_CODE = [
    'deliveries/__init__.py',
    'deliveries/deliveries_problem_input.py',
    'framework/__init__.py',
    'framework/consts.py',
    'framework/db/test1_big_delivery.in',
    'framework/db/test1_small_delivery.in',
    'framework/db/tlv.csv',
    'framework/graph_search/utils/__init__.py',
    'framework/graph_search/utils/heapdict.py',
    'framework/graph_search/utils/timer.py',
    'framework/graph_search/__init__.py',
    'framework/graph_search/best_first_search.py',
    'framework/graph_search/graph_problem_interface.py',
    'framework/graph_search/uniform_cost.py',
    'framework/ways/__init__.py',
    'framework/ways/draw.py',
    'framework/ways/graph.py',
    'framework/ways/tools.py'
]

TEST_SCRIPT_FILES = [
    'submission_test.py',
    'tests_utils/deliveries_tests_creator.py',
    'tests_utils/tests_utils.py',
    'tests_utils/tests_consts.py',
    'tests_utils/__init__.py'
]
TEST_SCRIPT_FILENAME = 'submission_test.py'
