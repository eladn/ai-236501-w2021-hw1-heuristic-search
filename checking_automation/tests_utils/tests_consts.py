import os

__all__ = [
    'PYTHON_INTERPRETER_FOR_SUBMISSIONS_TESTS',
    'SUBMISSIONS_PATH',
    'TESTS_ENVIRONMENTS_PATH',
    'TESTS_LOGS_PATH',
    'CLEAN_SUPPLIED_CODE_ENV_PATH',
    'STAFF_SOLUTION_CODE_PATH',
    'ADHOC_CODE_FIXES_FOR_CHECKING_PATH',
    'TEST_CALCULATED_TIMEOUTS_PATH',
    'TEST_STAFF_SOLUTION_TIMES_PATH',
    'STAFF_SOLUTION_DUMMY_ID',
    'NUMPY_RANDOM_SEED_BASE',
    'NR_ATTEMPTS_PER_TEST',
    'TEST_TIME_OVERHEAD_EST_IN_SECONDS',
    'DEFAULT_NR_PROCESSES',
    'DEFAULT_SUBMISSIONS_SAMPLE_SIZE',
    'CHECK_AUTOMATION_CODE_PATH',
    'NONVITAL_REQUIRED_SUBMISSION_CODE_FILES',
    'FILES_ASKED_NOT_TO_SUBMIT',
    'VITAL_REQUIRED_SUBMISSION_CODE_FILES',
    'FILES_TO_COPY_FROM_CLEAN_SUPPLIED_CODE',
    'TEST_SCRIPT_FILES',
    'TEST_SCRIPT_FILENAME',
]

PYTHON_INTERPRETER_FOR_SUBMISSIONS_TESTS = '/usr/local/bin/python3'
SUBMISSIONS_PATH = '/Users/eladn/ai-s2020-hw1-all-submissions-unzipped/'
TESTS_ENVIRONMENTS_PATH = '/Users/eladn/ai-s2020-hw1-submissions-checking/tests/'
TESTS_LOGS_PATH = '/Users/eladn/ai-s2020-hw1-submissions-checking/tests-logs/'
CLEAN_SUPPLIED_CODE_ENV_PATH = '../supplied_code/'
STAFF_SOLUTION_CODE_PATH = '../staff_solution/'
ADHOC_CODE_FIXES_FOR_CHECKING_PATH = '../adhoc_code_fixes_for_checking/'
CHECK_AUTOMATION_CODE_PATH = './'
TEST_CALCULATED_TIMEOUTS_PATH = os.path.join(TESTS_LOGS_PATH, 'tests-calculated-timeouts.txt')
TEST_STAFF_SOLUTION_TIMES_PATH = os.path.join(TESTS_LOGS_PATH, 'tests-staff-solution-times.txt')
STAFF_SOLUTION_DUMMY_ID = 1111118
NUMPY_RANDOM_SEED_BASE = 238435
NR_ATTEMPTS_PER_TEST = 3
TEST_TIME_OVERHEAD_EST_IN_SECONDS = 4  # for creating a process and load the roads
DEFAULT_NR_PROCESSES = 2
DEFAULT_SUBMISSIONS_SAMPLE_SIZE = 15

NONVITAL_REQUIRED_SUBMISSION_CODE_FILES = [
    'main.py'
]

FILES_ASKED_NOT_TO_SUBMIT = [
    'framework/db/tlv_streets_map.csv'
]

VITAL_REQUIRED_SUBMISSION_CODE_FILES = [
    'problems/cached_map_distance_finder.py',
    'problems/mda_heuristics.py',
    'problems/mda_problem.py',
    'problems/map_heuristics.py',
    'problems/map_problem.py',
    'framework/graph_search/astar.py',
    'framework/graph_search/astar_epsilon.py',
    'framework/graph_search/anytime_astar.py',
]

FILES_TO_COPY_FROM_CLEAN_SUPPLIED_CODE = [
    'problems/__init__.py',
    'problems/cached_air_distance_calculator.py',
    'problems/mda_problem_input.py',
    'framework/__init__.py',
    'framework/consts.py',
    'framework/serializable.py',
    'framework/db/test_mda_small_input.in',
    'framework/db/test_mda_medium_input.in',
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
    'tests_utils/mda_tests_creator.py',
    'tests_utils/tests_utils.py',
    'tests_utils/tests_consts.py',
    'tests_utils/__init__.py',
]
TEST_SCRIPT_FILENAME = 'submission_test.py'
