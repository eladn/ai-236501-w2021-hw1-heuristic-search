import os

__all__ = [
    'PYTHON_INTERPRETER_FOR_SUBMISSIONS_TESTS',
    'SUBMISSIONS_PATH',
    'TESTS_ENVIRONMENTS_PATH',
    'TESTS_LOGS_PATH',
    'CLEAN_SUPPLIED_CODE_ENV_PATH',
    'STAFF_SOLUTION_CODE_PATH',
    'TEST_CALCULATED_TIMEOUTS_PATH',
    'STAFF_SOLUTION_DUMMY_ID',
    'NUMPY_RANDOM_SEED_BASE',
    'NR_ATTEMPTS_PER_TEST',
    'TEST_TIME_OVERHEAD_EST_IN_SECONDS',
    'DEFAULT_NR_PROCESSES',
    'CHECK_AUTOMATION_CODE_PATH',
    'NONVITAL_REQUIRED_SUBMISSION_CODE_FILES',
    'FILES_ASKED_NOT_TO_SUBMIT',
    'VITAL_REQUIRED_SUBMISSION_CODE_FILES',
    'FILES_TO_COPY_FROM_CLEAN_SUPPLIED_CODE',
    'TEST_SCRIPT_FILES',
    'TEST_SCRIPT_FILENAME'
]

PYTHON_INTERPRETER_FOR_SUBMISSIONS_TESTS = '/usr/local/Cellar/python/3.7.6/bin/python3.7'
SUBMISSIONS_PATH = '/Users/eladn/Downloads/HW1-3240001574168428_236501_Winter2019-2020_unzipped'
TESTS_ENVIRONMENTS_PATH = '/Users/eladn/Documents/ai-w2020-hw1-submissions/tests/'
TESTS_LOGS_PATH = '/Users/eladn/Documents/ai-w2020-hw1-submissions/tests-logs/'
CLEAN_SUPPLIED_CODE_ENV_PATH = '../supplied_code/'
STAFF_SOLUTION_CODE_PATH = '../staff_solution/'
CHECK_AUTOMATION_CODE_PATH = './'
TEST_CALCULATED_TIMEOUTS_PATH = os.path.join(TESTS_LOGS_PATH, 'tests-calculated-timeouts.txt')
STAFF_SOLUTION_DUMMY_ID = 1111118
NUMPY_RANDOM_SEED_BASE = 238435
NR_ATTEMPTS_PER_TEST = 2
TEST_TIME_OVERHEAD_EST_IN_SECONDS = 4  # for creating a process and load the roads
DEFAULT_NR_PROCESSES = 2

NONVITAL_REQUIRED_SUBMISSION_CODE_FILES = [
    'main.py'
]

FILES_ASKED_NOT_TO_SUBMIT = [
    'framework/db/tlv_streets_map.csv'
]

VITAL_REQUIRED_SUBMISSION_CODE_FILES = [
    'deliveries/cached_map_distance_finder.py',
    'deliveries/deliveries_truck_heuristics.py',
    'deliveries/deliveries_truck_problem.py',
    'deliveries/map_heuristics.py',
    'deliveries/map_problem.py',
    'framework/graph_search/astar.py',
    'framework/graph_search/astar_epsilon.py',
    'framework/graph_search/anytime_astar.py',
]

FILES_TO_COPY_FROM_CLEAN_SUPPLIED_CODE = [
    'deliveries/__init__.py',
    'deliveries/cached_air_distance_calculator.py',
    'deliveries/deliveries_truck_problem_input.py',
    'framework/__init__.py',
    'framework/consts.py',
    'framework/db/test_deliveries_small.in',
    'framework/db/test_deliveries_medium.in',
    'framework/db/tlv_streets_map.csv',
    'framework/graph_search/utils/__init__.py',
    'framework/graph_search/utils/heapdict.py',
    'framework/graph_search/utils/timer.py',
    'framework/graph_search/utils/utils.py',
    'framework/graph_search/__init__.py',
    'framework/graph_search/best_first_search.py',
    'framework/graph_search/graph_problem_interface.py',
    'framework/graph_search/uniform_cost.py',
    'framework/ways/__init__.py',
    'framework/ways/streets_map.py',
]

TEST_SCRIPT_FILES = [
    'submission_test.py',
    'tests_utils/deliveries_tests_creator.py',
    'tests_utils/tests_utils.py',
    'tests_utils/tests_consts.py',
    'tests_utils/__init__.py',
]
TEST_SCRIPT_FILENAME = 'submission_test.py'
