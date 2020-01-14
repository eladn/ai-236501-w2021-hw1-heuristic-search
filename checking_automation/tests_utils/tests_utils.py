import os
import argparse
import sys
import shutil
from typing import *
from collections import OrderedDict
import subprocess
import time
import numpy as np
import importlib
import autopep8

from .tests_consts import *

__all__ = ['timeout_exec', 'TestResult', 'is_dir_contains_files', 'iterate_inner_directories',
           'make_dirs_if_not_exist', 'dyn_load_module', 'Submission', 'get_module_by_name',
           'HeuristicFactory', 'SolverFactory', 'ProblemFactory', 'SubmissionTest', 'SubmissionTestsSuit',
           'argparse_file_path_type', 'argparse_dir_path_type', 'copy_staff_solution_as_submission']


# I tried to limit the execution time for each test separately.
# For that end, I tried to use `timeout_exec()` which is implemented
# in `tests_utils.py`. I've tries different implementations for this
# mechanism. All implementations used threading. None of them worked.
# Using these methods always creates serious bugs - simple asserts fail.
# I believe it creates a race. It is strange because these threads
# are not executed in parallel - the main thread just waits for some
# worker thread to complete its execution. I couldn't understand
# what race does it make. Probably the Python interpreter is just
# not meant for concurrency.
# Now I do it using the timeout mechanism of `subprocess` package.
def timeout_exec(func, args=(), kwargs=None, timeout_duration=10, default=None):
    """This function will spawn a thread and run the given function
    using the args, kwargs and return the given default value if the
    timeout_duration is exceeded.
    """
    if kwargs is None:
        kwargs = {}
    import multiprocessing.pool
    # originally it was `ThreadPool` (rather than `Pool` which uses processes)
    pool = multiprocessing.pool.Pool(processes=1)
    async_result = pool.apply_async(func, args, kwargs)
    try:
        res = async_result.get(timeout_duration)
        return res
    except multiprocessing.TimeoutError as e:
        pool.close()
        return default
    except Exception as e:
        raise e


def is_dir_contains_files(dir_path: str, files: List[str]):
    return all(os.path.isfile(os.path.join(dir_path, filename)) for filename in files)


def iterate_inner_directories(dir_path: str, depth_limit: int = 3):
    yield ''
    if depth_limit <= 0:
        return
    for inner_dir in os.listdir(dir_path):
        if inner_dir[0] != '_' and os.path.isdir(os.path.join(dir_path, inner_dir)):
            for inner_inner_dir in iterate_inner_directories(os.path.join(dir_path, inner_dir), depth_limit-1):
                yield os.path.join(inner_dir, inner_inner_dir)


def make_dirs_if_not_exist(path, inner_dir):
    if not inner_dir.strip():
        return
    if os.path.exists(os.path.join(path, inner_dir)):
        return
    upper_level_dir = os.path.dirname(inner_dir)
    make_dirs_if_not_exist(path, upper_level_dir)
    os.mkdir(os.path.join(path, inner_dir))


def dyn_load_module(module_name: str, module_init_file_path: str, sys_module=None):
    if sys_module is None:
        sys_module = sys
    import importlib.util
    module_spec = importlib.util.spec_from_file_location(module_name, module_init_file_path)
    module = importlib.util.module_from_spec(module_spec)
    sys_module.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    assert module_name in globals()


def get_module_by_name(name: str):
    framework_module = importlib.import_module('framework')
    deliveries_module = importlib.import_module('deliveries')
    assert name in framework_module.__dict__ or name in deliveries_module.__dict__
    if name in framework_module.__dict__:
        return framework_module.__dict__[name]
    else:
        return deliveries_module.__dict__[name]


class HeuristicFactory(NamedTuple):
    name: str

    def get_heuristic_ctor(self):
        return get_module_by_name(self.name)


class SolverFactory(NamedTuple):
    name: str
    heuristic: Optional[HeuristicFactory] = None
    params: Tuple = ()
    ctor_kwargs: Dict = {}

    def create_instance(self):
        framework_module = importlib.import_module('framework')
        solver_ctor = framework_module.__dict__[self.name]
        use_heuristic = self.name == 'AStar' or self.name == 'AnytimeAStar' or self.name == 'AStarEpsilon'
        assert not use_heuristic or self.heuristic is not None
        assert use_heuristic or self.heuristic is None
        solver_ctor_args = tuple(self.params)
        if use_heuristic:
            heuristic_ctor = self.heuristic.get_heuristic_ctor()
            solver_ctor_args = (heuristic_ctor,) + solver_ctor_args
        solver_instance = solver_ctor(*solver_ctor_args, **self.ctor_kwargs)
        return solver_instance

    def get_full_name(self):
        all_params = ()
        if self.heuristic:
            all_params = all_params + (self.heuristic.name, )
        all_params = all_params + self.params
        params_str = ''
        all_params = [str(param) for param in all_params]
        if len(all_params) > 0:
            params_str = '<' + ('; '.join(all_params)) + '>'
        return self.name + params_str


class ProblemFactory(NamedTuple):
    name: str
    input_name: Optional[str] = None
    params: Tuple = ()
    kwargs_builder: Dict = {}
    inner_problem_solver: Optional[SolverFactory] = None

    def create_instance(self, roads):
        deliveries_module = importlib.import_module('deliveries')
        problem_ctor = deliveries_module.__dict__[self.name]
        use_problem_input = self.name == 'DeliveriesTruckProblem'
        use_roads_in_problem_ctor = self.name == 'MapProblem' or self.name == 'DeliveriesTruckProblem'
        assert not use_problem_input or self.input_name is not None
        assert use_problem_input or self.input_name is None
        problem_ctor_args = tuple(self.params)
        problem_ctor_kwargs = {key: value_ctor() for key, value_ctor in self.kwargs_builder.items()}
        if self.inner_problem_solver is not None:
            inner_problem_solver_instance = self.inner_problem_solver.create_instance()
            # problem_ctor_kwargs['inner_problem_solver'] = inner_problem_solver_instance
            problem_ctor_args = (inner_problem_solver_instance,) + problem_ctor_args
        if use_roads_in_problem_ctor:
            problem_ctor_args = (roads,) + problem_ctor_args
        if use_problem_input:
            DeliveriesProblemInput = deliveries_module.__dict__['DeliveriesTruckProblemInput']
            problem_input = DeliveriesProblemInput.load_from_file(self.input_name + '.in', roads)
            problem_ctor_args = (problem_input,) + problem_ctor_args
        problem_instance = problem_ctor(*problem_ctor_args, **problem_ctor_kwargs)
        return problem_instance

    def get_full_name(self):
        all_params = ()

        if self.input_name is not None:
            all_params = all_params + ('input=' + self.input_name, )
        if self.inner_problem_solver is not None:
            all_params = all_params + ('inner_problem_solver=' + self.inner_problem_solver.get_full_name(), )
        all_params = all_params + self.params

        params_str = ''
        all_params = [str(param) for param in all_params]
        if len(all_params) > 0:
            params_str = '<' + ('; '.join(all_params)) + '>'

        return self.name + params_str


class SubmissionTest(NamedTuple):
    name: str
    index: int
    problem_factory: ProblemFactory
    solver_factory: SolverFactory
    execution_timeout: int
    execute_in_submission_test_env: bool = True
    files_to_override_from_staff_solution: Tuple[str] = ()
    files_to_override_from_adhoc_code_fixes: Tuple[str] = ()
    fn_to_execute_before_solving: Optional[Callable] = None

    def run_test(self, roads, store_execution_log: bool = False):
        self.init_numpy_seed()
        problem_instance = self.problem_factory.create_instance(roads)
        solver_instance = self.solver_factory.create_instance()
        if self.fn_to_execute_before_solving is not None:
            self.fn_to_execute_before_solving(problem_instance, solver_instance)
        if store_execution_log:
            solver_instance.store_execution_log = True
        res = solver_instance.solve_problem(problem_instance)
        execution_log = solver_instance.last_execution_log.to_log_format() if store_execution_log else None
        return TestResult.from_search_result(res), res, execution_log

    def calc_seed(self) -> int:
        return hash((NUMPY_RANDOM_SEED_BASE + self.index, self.index)) % (2 ** 32)

    def init_numpy_seed(self):
        # Make `np.random` behave deterministic.
        numpy_random_seed = self.calc_seed()
        np.random.seed(numpy_random_seed)

    def get_full_name(self) -> str:
        return '{idx} :: {test_name} :: {solver_name}({problem_name})'.format(
            idx=self.index,
            test_name=self.name,
            solver_name=self.solver_factory.get_full_name(),
            problem_name=self.problem_factory.get_full_name()
        )


class TestResult(NamedTuple):
    dev: int
    space: int
    cost: Optional[float]
    path: Tuple[int]

    @staticmethod
    def from_search_result(search_result):
        # assert isinstance(search_result, SearchResult)
        if not search_result.is_solution_found:
            return TestResult(dev=search_result.nr_expanded_states, space=search_result.max_nr_stored_states, cost=None, path=())
        path = tuple(state.state.current_location.index if hasattr(state.state, 'current_location') else state.state.junction_id for state in search_result.solution_path)
        return TestResult(dev=search_result.nr_expanded_states, space=search_result.max_nr_stored_states, cost=search_result.solution_g_cost, path=path)

    def serialize(self) -> str:
        return f'dev:{self.dev} -- space:{self.space} -- cost:{self.cost} -- path:{self.path}'

    @staticmethod
    def deserialize(serialized: str) -> Optional['TestResult']:
        import regex
        positive_integer_pattern = '(([1-9][0-9]*)|0)'
        list_of_positive_integers_pattern = '(((' + positive_integer_pattern + '([\\ ]*,[\\ ]*))*)' + positive_integer_pattern + ')'
        positive_float_pattern = '(' + positive_integer_pattern + '(\\.[0-9]*)?)'
        dev_pattern = 'dev:(?P<dev>' + positive_integer_pattern + ')'
        space_pattern = 'space:(?P<space>' + positive_integer_pattern + ')'
        cost_pattern = 'cost:(?P<cost>' + positive_float_pattern + ')'
        path_pattern = 'path:\\((?P<path>' + list_of_positive_integers_pattern + ')\\)'
        test_result_pattern = '^' + ('[\\ ]+\\-\\-[\\ ]+'.join([dev_pattern, space_pattern, cost_pattern, path_pattern])) + '$'

        serialized = serialized.strip()
        test_result_parser = regex.compile(test_result_pattern)
        parsed_test_result = test_result_parser.match(serialized)

        try:
            assert parsed_test_result
            assert len(parsed_test_result.capturesdict()['dev']) == 1
            dev = int(parsed_test_result.capturesdict()['dev'][0])
            assert len(parsed_test_result.capturesdict()['space']) == 1
            space = int(parsed_test_result.capturesdict()['space'][0])
            assert len(parsed_test_result.capturesdict()['cost']) == 1
            cost = float(parsed_test_result.capturesdict()['cost'][0])
            assert len(parsed_test_result.capturesdict()['path']) == 1
            path = tuple(int(s) for s in parsed_test_result.capturesdict()['path'][0].split(','))
            return TestResult(dev=dev, space=space, cost=cost, path=path)
        except:
            return None

    @staticmethod
    def load_from_file(tests_result_file_path: str) -> Optional['TestResult']:
        if not os.path.isfile(tests_result_file_path):
            return None
        with open(tests_result_file_path, 'r') as result_file:
            tests_results = [
                TestResult.deserialize(line)
                for line in result_file.readlines()
                if line.strip() != ''
            ]
            assert len(tests_results) <= 1
            if len(tests_results) < 1:
                return None
            return tests_results[0]

    # TODO: remove this method.
    @staticmethod
    def read_tests_results_file(tests_results_file_path: str) -> Optional[List['TestResult']]:
        if not os.path.isfile(tests_results_file_path):
            return None
        with open(tests_results_file_path, 'r') as results_file:
            tests_results = [
                TestResult.deserialize(line)
                for line in results_file.readlines()
                if line.strip() != ''
            ]
            return tests_results

    def __eq__(self, other):
        assert isinstance(other, TestResult)
        return self.dev == other.dev \
               and abs(self.cost - other.cost) < 0.0001 \
               and len(self.path) == len(other.path) \
               and all(s1 == s2 for s1, s2 in zip(self.path, other.path))

    def is_acceptable(self, staff_test_result: 'TestResult'):
        assert isinstance(staff_test_result, TestResult)
        return max(self.dev, staff_test_result.dev) <= min(self.dev, staff_test_result.dev) * 1.15 \
               and abs(self.cost - staff_test_result.cost) < 0.01 \
               and len(self.path) == len(staff_test_result.path) \
               and all(s1 == s2 for s1, s2 in zip(self.path, staff_test_result.path))


class SubmissionTestsSuit:
    def __init__(self):
        self._tests_by_idx_mapping: OrderedDict[int, SubmissionTest] = OrderedDict()

    def __iter__(self):
        for test in self._tests_by_idx_mapping.values():
            yield test

    def __len__(self):
        return len(self._tests_by_idx_mapping)

    def get_test_by_idx(self, test_idx: int):
        return self._tests_by_idx_mapping[test_idx]

    def filter_tests_by_idx(self, tests_indices: Union[List[int], Tuple[int]]) -> 'SubmissionTestsSuit':
        new_tests_suit = SubmissionTestsSuit()
        for idx in tests_indices:
            new_tests_suit._tests_by_idx_mapping[idx] = self.get_test_by_idx(idx)
        return new_tests_suit

    def create_test(self, **kwargs):
        if 'index' not in kwargs:
            new_test_idx = len(self._tests_by_idx_mapping)
            kwargs = {'index': new_test_idx, **kwargs}
        assert kwargs['index'] not in self._tests_by_idx_mapping.keys()
        new_test = SubmissionTest(**kwargs)
        self._tests_by_idx_mapping[new_test.index] = new_test

    def calc_overall_tests_execution_time(self) -> int:
        return sum(TEST_TIME_OVERHEAD_EST_IN_SECONDS + test.execution_timeout for test in self._tests_by_idx_mapping.values())

    def get_tests_names(self) -> List[str]:
        return [test.get_full_name() for test in self._tests_by_idx_mapping.values()]

    def create_astar_tests_for_weights_in_range(
            self,
            heuristic_name: str,
            problem_factory: ProblemFactory,
            execution_timeout: Union[int, Tuple[int, int]],
            n: int = 7):
        weights = np.linspace(0.5, 1, n)
        if isinstance(execution_timeout, tuple):
            execution_timeouts = np.linspace(execution_timeout[0], execution_timeout[1], n)
        else:
            execution_timeouts = np.ones(n) * execution_timeout
        for w, cur_execution_timeout in zip(weights, execution_timeouts):
            astar_importer = SolverFactory(name='AStar', heuristic_name=heuristic_name, params=(w,))
            self.create_test(
                problem_factory=problem_factory,
                solver_factory=astar_importer,
                execution_timeout=cur_execution_timeout)

    def update_timeouts(self, new_timeouts_per_test):
        for test_idx, new_timeout in new_timeouts_per_test.items():
            if test_idx not in self._tests_by_idx_mapping:
                continue
            old_test = self._tests_by_idx_mapping[test_idx]
            dct = old_test._asdict()
            dct['execution_timeout'] = new_timeout
            new_test = type(old_test)(**dct)
            self._tests_by_idx_mapping[test_idx] = new_test


class Submission:
    main_submission_path: str
    main_submission_directory_name: str
    ids: Tuple[int]
    code_dirs: List[str]
    code_dir: Optional[str]
    code_path: Optional[str]
    followed_assignment_instructions: bool
    tests_environment_path: str

    def __init__(self, main_submission_path: str):
        self.main_submission_path = main_submission_path
        self.main_submission_directory_name = os.path.basename(main_submission_path.rstrip('/').rstrip('\\'))
        self.ids = tuple(int(stud_id) for stud_id in self.main_submission_directory_name.split('-'))
        self.code_dirs = self.find_code_dirs()
        self.code_dir = self.code_dirs[0] if len(self.code_dirs) > 0 else None
        self.code_path = os.path.join(self.main_submission_path, self.code_dir) if self.code_dir is not None else None
        self.followed_assignment_instructions = self.check_if_followed_assignment_instructions()
        self.tests_environment_path = os.path.join(TESTS_ENVIRONMENTS_PATH, self.main_submission_directory_name)
        self.tests_logs_dir_path = os.path.join(TESTS_LOGS_PATH, self.main_submission_directory_name)

    def check_if_followed_assignment_instructions(self):
        return self.code_dir == 'ai_hw1' and \
               is_dir_contains_files(self.code_path, VITAL_REQUIRED_SUBMISSION_CODE_FILES) and \
               is_dir_contains_files(self.code_path, NONVITAL_REQUIRED_SUBMISSION_CODE_FILES) and \
               not is_dir_contains_files(self.code_path, FILES_ASKED_NOT_TO_SUBMIT)

    def find_code_dirs(self):
        return [
                code_dir.rstrip('/').rstrip('\\')
                for code_dir in iterate_inner_directories(self.main_submission_path, depth_limit=3)
                if is_dir_contains_files(os.path.join(self.main_submission_path, code_dir),
                                         VITAL_REQUIRED_SUBMISSION_CODE_FILES)
            ]

    def make_clean_tests_environment(self, tests_suit: SubmissionTestsSuit, tests_environment_path: str = None, override_if_exists: bool = True):
        if tests_environment_path is None:
            tests_environment_path = self.tests_environment_path
        if os.path.exists(tests_environment_path):
            if not override_if_exists:
                return
            shutil.rmtree(tests_environment_path)
        os.mkdir(tests_environment_path)

        if os.path.exists(self.tests_logs_dir_path):
            shutil.rmtree(self.tests_logs_dir_path)
        os.mkdir(self.tests_logs_dir_path)

        for test in tests_suit:
            test_environment_path_path = os.path.join(tests_environment_path, f'test-{test.index}')
            os.mkdir(test_environment_path_path)
            files_to_override_from_staff_solution = set()
            files_to_override_from_adhoc_code_fixes = set()
            if test.files_to_override_from_staff_solution is not None:
                files_to_override_from_staff_solution = set(test.files_to_override_from_staff_solution)
            if test.files_to_override_from_adhoc_code_fixes is not None:
                files_to_override_from_adhoc_code_fixes = set(test.files_to_override_from_adhoc_code_fixes)
            files_not_to_copy_from_submission_or_clean_supplied_code = set(files_to_override_from_staff_solution) | \
                                                                       set(files_to_override_from_adhoc_code_fixes)
            for submitted_file in set(VITAL_REQUIRED_SUBMISSION_CODE_FILES) - set(files_not_to_copy_from_submission_or_clean_supplied_code):
                make_dirs_if_not_exist(test_environment_path_path, os.path.dirname(submitted_file))
                shutil.copy2(os.path.join(self.code_path, submitted_file),
                             os.path.join(test_environment_path_path, os.path.dirname(submitted_file)))
                if submitted_file.split('.')[-1] == 'py':
                    filepath = os.path.join(test_environment_path_path, submitted_file)
                    autopep8.fix_file(filepath, options=autopep8.parse_args([filepath, '-i']))
            for submitted_file in set(NONVITAL_REQUIRED_SUBMISSION_CODE_FILES) - set(files_not_to_copy_from_submission_or_clean_supplied_code):
                if not os.path.isfile(os.path.join(self.code_path, submitted_file)):
                    continue
                make_dirs_if_not_exist(test_environment_path_path, os.path.dirname(submitted_file))
                shutil.copy2(os.path.join(self.code_path, submitted_file),
                             os.path.join(test_environment_path_path, os.path.dirname(submitted_file)))
                if submitted_file.split('.')[-1] == 'py':
                    filepath = os.path.join(test_environment_path_path, submitted_file)
                    autopep8.fix_file(filepath, options=autopep8.parse_args([filepath, '-i']))
            for file in set(FILES_TO_COPY_FROM_CLEAN_SUPPLIED_CODE) - set(files_not_to_copy_from_submission_or_clean_supplied_code):
                make_dirs_if_not_exist(test_environment_path_path, os.path.dirname(file))
                shutil.copy2(os.path.join(CLEAN_SUPPLIED_CODE_ENV_PATH, file),
                             os.path.join(test_environment_path_path, os.path.dirname(file)))
            for test_file in set(TEST_SCRIPT_FILES):
                make_dirs_if_not_exist(test_environment_path_path, os.path.dirname(test_file))
                shutil.copy2(os.path.join(CHECK_AUTOMATION_CODE_PATH, test_file),
                             os.path.join(test_environment_path_path, os.path.dirname(test_file)))
            for staff_solution_file in files_to_override_from_staff_solution:
                make_dirs_if_not_exist(test_environment_path_path, os.path.dirname(staff_solution_file))
                shutil.copy2(os.path.join(STAFF_SOLUTION_CODE_PATH, staff_solution_file),
                             os.path.join(test_environment_path_path, os.path.dirname(staff_solution_file)))
            for adhoc_code_fix_file in files_to_override_from_adhoc_code_fixes:
                make_dirs_if_not_exist(test_environment_path_path, os.path.dirname(adhoc_code_fix_file))
                shutil.copy2(os.path.join(ADHOC_CODE_FIXES_FOR_CHECKING_PATH, adhoc_code_fix_file),
                             os.path.join(test_environment_path_path, os.path.dirname(adhoc_code_fix_file)))

    def run_tests_suit_in_tests_environment(
            self, tests_indices: Tuple[int], store_execution_log: bool = False) -> Dict[int, float]:
        from .deliveries_tests_creator import DeliveriesTestsSuitCreator
        tests_suit = DeliveriesTestsSuitCreator.create_tests_suit()
        tests_suit = tests_suit.filter_tests_by_idx(tests_indices)

        self.make_clean_tests_environment(tests_suit)

        execution_time_per_test = {}

        class TestExecAttempt(NamedTuple):
            test: SubmissionTest
            attempt_nr: int

        tests_to_run = [
            TestExecAttempt(test=test, attempt_nr=1)
            for test in tests_suit if test.execute_in_submission_test_env
        ]
        tests_to_run.reverse()
        while len(tests_to_run) > 0:
            cur_test_to_run, cur_test_attempt_nr = tests_to_run.pop()
            test_str = f'test-{cur_test_to_run.index}'
            test_out_path = os.path.join(self.tests_logs_dir_path, test_str)

            # files_to_override_from_staff_solution

            output_files_exts = ['res', 'out', 'err', 'exec_time', 'exec_log']
            for ext in output_files_exts:
                file_to_delete = f'{test_out_path}.{ext}'
                if os.path.isfile(file_to_delete):
                    os.remove(file_to_delete)

            run_args = [
                PYTHON_INTERPRETER_FOR_SUBMISSIONS_TESTS,
                TEST_SCRIPT_FILENAME,
                f'{test_out_path}.res',
                str(cur_test_to_run.index)
            ]
            if store_execution_log:
                run_args.append(f'{test_out_path}.exec_log')
            timeout = cur_test_to_run.execution_timeout
            if store_execution_log:
                # It takes time just to store the execution log.
                # TODO [staff]: maybe we would like another mechanism to limit the timeout here.
                timeout *= 10

            test_environment_path_path = os.path.join(self.tests_environment_path, f'test-{cur_test_to_run.index}')
            start_time = time.time()
            try:
                submission_test_process = subprocess.run(
                    run_args,
                    cwd=test_environment_path_path,
                    capture_output=True,
                    timeout=timeout
                )
                end_time = time.time()
                exec_time = end_time - start_time
                execution_time_per_test[cur_test_to_run.index] = exec_time
                with open(test_out_path + '.out', 'wb') as test_run_stdout:
                    test_run_stdout.write(submission_test_process.stdout)
                with open(test_out_path + '.err', 'wb') as test_run_stderr:
                    test_run_stderr.write(submission_test_process.stderr)
                with open(test_out_path + '.exec_time', 'w') as test_exec_time_file:
                    test_exec_time_file.write('execution time: {} sec.\n'.format(exec_time))
            except subprocess.TimeoutExpired:
                end_time = time.time()
                exec_time = end_time - start_time
                execution_time_per_test[cur_test_to_run.index] = exec_time
                with open(test_out_path + '.err', 'w') as test_run_stderr:
                    test_run_stderr.write('error: timeout reached during tests subprocess execution.\n')
                with open(test_out_path + '.exec_time', 'w') as test_exec_time_file:
                    test_exec_time_file.write('execution time: {} sec.\n'.format(exec_time))
                if cur_test_attempt_nr < NR_ATTEMPTS_PER_TEST:
                    tests_to_run.insert(0, TestExecAttempt(test=cur_test_to_run, attempt_nr=(cur_test_attempt_nr + 1)))
                time.sleep(0.5)
        return execution_time_per_test

    def _load_modules_from_test_env(self):
        """Not used."""
        self.make_clean_tests_environment()
        assert os.getcwd() in sys.path
        sys.path.remove(os.getcwd())
        sys.path.append(self.tests_environment_path)
        os.chdir(self.tests_environment_path)

        try:
            framework_path = os.path.join(self.tests_environment_path, "framework/__init__.py")
            deliveries_path = os.path.join(self.tests_environment_path, "deliveries/__init__.py")

            # make `os`, `sys` packages unusable.
            # local_modules = dict(sys.modules)
            # local_sys_module = sys
            # del os
            # sys.modules['os'] = None
            # sys.modules['sys'] = None
            # del sys

            dyn_load_module("framework", framework_path)
            dyn_load_module("deliveries", deliveries_path)

            # from framework import *
            # from deliveries import *

        except:
            pass  # TODO

    @staticmethod
    def load_all_submissions(ids_filter: Optional[Collection[int]] = None) -> List['Submission']:
        if ids_filter is not None:
            ids_filter = {int(identifier) for identifier in ids_filter}
        copy_staff_solution_as_submission(STAFF_SOLUTION_DUMMY_ID)
        all_submissions_dirs = [
            os.path.join(SUBMISSIONS_PATH, submission_directory)
            for submission_directory in os.listdir(SUBMISSIONS_PATH)
            if os.path.isdir(os.path.join(SUBMISSIONS_PATH, submission_directory)) \
               and all(identifier.isdigit() for identifier in submission_directory.split('-')) \
               and (ids_filter is None or any(
                int(identifier) in ids_filter for identifier in submission_directory.split('-')))]
        all_submissions = [Submission(submission_dir) for submission_dir in all_submissions_dirs]
        return all_submissions

    def load_tests_results(self, tests_suit: SubmissionTestsSuit) -> 'SubmissionTestsResults':
        all_results = []
        for test in tests_suit:
            test_result_filename = 'test-{idx}.res'.format(idx=test.index)
            test_result_file_path = os.path.join(self.tests_logs_dir_path, test_result_filename)
            test_result = TestResult.load_from_file(test_result_file_path)
            all_results.append(test_result)
        return SubmissionTestsResults(all_results)

    def remove_all_files(self):
        shutil.rmtree(self.main_submission_path)
        if os.path.isdir(self.tests_logs_dir_path):
            shutil.rmtree(self.tests_logs_dir_path)
        if os.path.isdir(self.tests_environment_path):
            shutil.rmtree(self.tests_environment_path)


class SubmissionTestsResults(List[Optional[TestResult]]):
    def __init__(self, tests_results: List[Optional[TestResult]]):
        super(SubmissionTestsResults, self).__init__(tests_results)
        self._pass_vector = None

    def calc_pass_vector(self, staff_solution_tests_results: 'SubmissionTestsResults'):
        assert len(staff_solution_tests_results) == len(self)
        self._pass_vector = np.array(list(
             int(isinstance(test_result, TestResult) and test_result.is_acceptable(staff_test_result))
             for test_result, staff_test_result in zip(self, staff_solution_tests_results)
        ), dtype=np.int)

    @property
    def pass_vector(self):
        assert self._pass_vector is not None
        return self._pass_vector


def argparse_file_path_type(input_path: str):
    input_path = str(input_path)
    if not os.path.isfile(input_path):
        raise argparse.ArgumentError(f'Given `{input_path}` is not a valid path of an existing file.')
    return input_path


def argparse_dir_path_type(input_path: str):
    input_path = str(input_path)
    if not os.path.isdir(input_path):
        raise argparse.ArgumentError(f'Given `{input_path}` is not a valid path of an existing file.')
    return input_path

def copy_staff_solution_as_submission(submission_id: int) -> Submission:
    staff_solution_submission_path = os.path.join(SUBMISSIONS_PATH, str(submission_id))
    if os.path.isdir(staff_solution_submission_path):
        shutil.rmtree(staff_solution_submission_path)
    shutil.copytree(STAFF_SOLUTION_CODE_PATH, staff_solution_submission_path)
    return Submission(staff_solution_submission_path)
