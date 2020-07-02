import math
import json
import os
import argparse
import dataclasses
import time
from typing import *
from functools import partial
from tests_utils import *
#from checking_automation.tests_utils import *
import numpy as np
from pprint import pprint

"""

Goes throughout submitted directories. For each submission:
1. find the path the relevant python files in the submission dir.
2. if not consistent with the assignment instruction - mark the `followed-assignment-instruction-test` as failed.
3. open new directory for the tests.
4. copy our files to this new directory.
5. copy the relevant submitted files to this directory.
6. run tests from this dir in dedicated processes.
   use processes from process pool.
   make these processes fail after timeouts.

"""


@dataclasses.dataclass
class JobsStatus:
    total_nr_jobs: int = 0
    nr_successfully_completed_jobs: int = 0
    nr_failed_jobs: int = 0
    start_time: Optional[float] = None
    last_completed_or_failed_job_time: Optional[float] = None

    def print_progress(self):
        print(f'{self.nr_finished_jobs}/{self.total_nr_jobs} jobs finished. '
              f'{self.nr_successfully_completed_jobs} completed successfully. '
              f'{self.nr_failed_jobs} failed. {self.nr_remaining_jobs} remain.')
        print(f'Running time until last completed job: {(self.running_time_until_last_completed_job / 60):.2f} min. '
              f'Avg time per job: {(self.avg_time_per_job / 60):.2f} min. '
              f'Estimated remaining time: {(self.avg_time_per_job * self.nr_remaining_jobs / 60):.2f} min.')

    @property
    def nr_finished_jobs(self) -> int:
        return self.nr_successfully_completed_jobs + self.nr_failed_jobs

    @property
    def nr_remaining_jobs(self) -> int:
        return self.total_nr_jobs - self.nr_finished_jobs

    @property
    def running_time_until_last_completed_job(self) -> float:
        return self.last_completed_or_failed_job_time - self.start_time

    @property
    def avg_time_per_job(self) -> float:
        return self.running_time_until_last_completed_job / self.nr_finished_jobs


def submission_tests_invoker(
        submission: Submission, tests_indices: Tuple[int], store_execution_log: bool = False) -> Dict[int, float]:
    return submission.run_tests_suit_in_tests_environment(tests_indices, store_execution_log=store_execution_log)


def run_tests_for_all_submissions(
        tests_suit: SubmissionTestsSuit,
        all_submissions: List[Submission],
        use_processes_pool: bool = True,
        store_execution_log: bool = False,
        nr_processes: int = DEFAULT_NR_PROCESSES) -> Dict[Tuple[int], Dict[int, float]]:
    tests_suit_overall_exec_time = tests_suit.calc_overall_tests_execution_time()

    print('Tests suit contains {nr_tests} tests. Total execution time: {low_exec_time:.2f} - {high_exec_time:.2f} min (per submission).'.format(
        nr_tests=len(tests_suit),
        low_exec_time=tests_suit_overall_exec_time / 60,
        high_exec_time=NR_ATTEMPTS_PER_TEST*tests_suit_overall_exec_time / 60
    ))

    print(f'Running tests on {len(all_submissions)} submissions.')
    if use_processes_pool:
        print(f'Using pool of {nr_processes} processes.')
    else:
        print('Run all tests on the main process.')

    from multiprocessing import Pool
    process_pool = Pool(nr_processes)
    jobs_status = JobsStatus()
    failed_submissions = []
    tests_execution_times_per_submission = {}

    def submission_tests_invoker__on_success(submission: Submission, returned_value):
        jobs_status.last_completed_or_failed_job_time = time.time()
        jobs_status.nr_successfully_completed_jobs += 1
        print('========   Successfully completed running tests for submission: ids={ids}   ========'.format(ids=submission.ids))
        jobs_status.print_progress()
        print()
        tests_execution_times_per_submission[tuple(submission.ids)] = returned_value

    def submission_tests_invoker__on_error(submission: Submission, error):
        print()
        print('XXXXXXXX   FAILED running tests for submission: ids={ids}   XXXXXXXX'.format(ids=submission.ids))
        print(error)
        print()
        failed_submissions.append(submission)
        jobs_status.last_completed_or_failed_job_time = time.time()
        jobs_status.nr_failed_jobs += 1
        jobs_status.print_progress()
        with open(os.path.join(submission.tests_logs_dir_path, 'test-run-stderr.txt'), 'w') as test_run_stderr:
            test_run_stderr.write(str(error))

    jobs_status.start_time = time.time()
    tests_indices = tuple(test.index for test in tests_suit)
    for submission in all_submissions:
        if use_processes_pool:
            print('Spawning tests worker for submission: ids: {ids} -- submission-dir: {submission_dir}'.format(
                ids=submission.ids,
                submission_dir=submission.main_submission_directory_name
            ))
            jobs_status.total_nr_jobs += 1
            process_pool.apply_async(
                submission_tests_invoker, (submission, tests_indices, store_execution_log),
                callback=partial(submission_tests_invoker__on_success, submission),
                error_callback=partial(submission_tests_invoker__on_error, submission))
        else:
            print('Running tests for submission: ids: {ids} -- submission-dir: {submission_dir}'.format(
                ids=submission.ids,
                submission_dir=submission.main_submission_directory_name
            ))
            submission_tests_invoker(submission, tests_indices, store_execution_log)
    process_pool.close()
    process_pool.join()

    print('Completed running tests for all submissions.')
    if failed_submissions:
        print('failed:')
        print(list(failed_submission.ids for failed_submission in failed_submissions))

    return tests_execution_times_per_submission


def update_tests_suit_timeout_limit_wrt_staff_solution_time(
        tests_suit: SubmissionTestsSuit,
        store_execution_log: bool = False,
        test_exec_timeout_limit_factor: Union[float, Dict[float, float]] = 2,
        nr_processes: int = DEFAULT_NR_PROCESSES):

    nr_staff_solution_executions = nr_processes * 3

    print(f'Running the tests suit over the staff solution for {nr_staff_solution_executions} times '
          'in order to calculate the execution timeout limit for each test.')
    print('The timeout calculation for each test takes into account the maximum time between these executions.')

    new_staff_solution_submissions = [
        copy_staff_solution_as_submission(int(str(STAFF_SOLUTION_DUMMY_ID) + str(execution_idx)))
        for execution_idx in range(nr_staff_solution_executions)
    ]
    results_for_all_staff_submissions = run_tests_for_all_submissions(
        tests_suit, new_staff_solution_submissions, store_execution_log=store_execution_log,
        nr_processes=nr_processes)
    print()

    if isinstance(test_exec_timeout_limit_factor, dict):
        test_exec_timeout_limit_factor = list(test_exec_timeout_limit_factor.items())
        test_exec_timeout_limit_factor.sort(key=lambda x: x[0])

    def factor_time(exec_time: float) -> float:
        if isinstance(test_exec_timeout_limit_factor, float):
            return exec_time * test_exec_timeout_limit_factor
        assert isinstance(test_exec_timeout_limit_factor, list)
        factored_sum = 0.0
        already_summed_time = 0.0
        for cur_step_high_bound, cur_step_factor in test_exec_timeout_limit_factor:
            remained_time = exec_time - already_summed_time
            cur_step_max_width = cur_step_high_bound - already_summed_time
            assert cur_step_max_width >= 0
            cur_step_actual_width = min(remained_time, cur_step_max_width)
            factored_sum += cur_step_actual_width * cur_step_factor
            already_summed_time += cur_step_actual_width
        return math.ceil(factored_sum)

    staff_solution_time_per_test = {
        test.index:
            max(submission_result[test.index]
                for submission_result in results_for_all_staff_submissions.values())
        for test in tests_suit
    }

    new_time_per_test = {
        test_idx: factor_time(staff_solution_time)
        for test_idx, staff_solution_time in staff_solution_time_per_test.items()
    }

    with open(os.path.join(TESTS_LOGS_PATH, 'tests-calculated-timeouts-readable.txt'), 'w') as calculated_timeouts_file:
        for test in tests_suit:
            calculated_timeouts_file.write(test.get_full_name())
            calculated_timeouts_file.write('\n')
            calculated_timeouts_file.write(f'execution timeout limit: {new_time_per_test[test.index]:.4f}\n\n')

    with open(TEST_CALCULATED_TIMEOUTS_PATH, 'w') as calculated_timeouts_file:
        for test in tests_suit:
            calculated_timeouts_file.write(f'{test.index}\n')
            calculated_timeouts_file.write(f'{new_time_per_test[test.index]}\n')

    compared_times_per_test = {
        test.index: {
            'orig': test.execution_timeout,
            'staff': staff_solution_time_per_test[test.index],
            'new': new_time_per_test[test.index]}
        for test in tests_suit
    }

    with open(TEST_STAFF_SOLUTION_TIMES_PATH, 'w') as test_staff_solution_times_file:
        test_staff_solution_times_file.write(json.dumps(compared_times_per_test))
        test_staff_solution_times_file.write('\n')

    print('Compared new-vs-old tests execution timeout limitation: ')
    pprint({tests_suit.get_test_by_idx(test_idx).get_full_name(): times
            for test_idx, times in compared_times_per_test.items()})
    print()

    tests_suit.update_timeouts(new_time_per_test)

    for staff_submission in new_staff_solution_submissions:
        staff_submission.remove_all_files()


def load_tests_suit_timeout_limit_from_stored_file(tests_suit: SubmissionTestsSuit):
    loaded_timeout_per_test: Dict[int, int] = {}
    if not os.path.exists(TEST_CALCULATED_TIMEOUTS_PATH):
        print(f'Cannot load tests-suit timeout limit. No such file `{TEST_CALCULATED_TIMEOUTS_PATH}`.')
        return
    with open(TEST_CALCULATED_TIMEOUTS_PATH, 'r') as calculated_timeouts_file:
        while True:
            test_idx_as_str = calculated_timeouts_file.readline()
            if test_idx_as_str == '' or test_idx_as_str is None:
                break  # EOF
            test_idx = int(test_idx_as_str)
            loaded_timeout_per_test[test_idx] = int(calculated_timeouts_file.readline())
    tests_suit.update_timeouts(loaded_timeout_per_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-idxs", dest="tests_idxs", type=int, nargs='+', required=False,
                        help="Tests indices to use (if not specified runs all tests in tests suit)")
    parser.add_argument("--submissions-ids", dest="submissions_ids", type=int, nargs='+', required=False,
                        help="IDs of submissions to check (if not specified runs on all submissions in dir)")
    parser.add_argument("--store-execution-log", dest='store_execution_log', action='store_true',
                        default=False, help='Whether to store the execution log')
    parser.add_argument("--sample-submissions", dest='sample_submissions',
                        nargs='?', required=False, type=float, default=False,
                        help='Execute on a sample of the submissions (argument may be the sample size/rate)')
    parser.add_argument("--update-tests-timeout", dest='update_tests_timeout', action='store_true',
                        help='Update tests suit timeout limit wrt staff solution time')
    parser.add_argument("--no-use-processes-pool", dest='no_use_processes_pool', action='store_true',
                        help='Run tests on current process without using a processes pool')
    parser.add_argument("--nr-processes", dest='nr_processes', default=DEFAULT_NR_PROCESSES, type=int,
                        help='Number of processes in the pool')
    args = parser.parse_args()

    tests_suit = MDATestsSuitCreator.create_tests_suit()

    os.makedirs(TESTS_LOGS_PATH, exist_ok=True)
    with open(os.path.join(TESTS_LOGS_PATH, 'test_names.txt'), 'w') as test_names_file:
        for test in tests_suit:
            test_names_file.write(f'{test.get_full_name()}\n')

    # tests_suit = tests_suit.filter_tests_by_idx([1, 13])

    # [optional] args may contain a list of test indices to run (otherwise run all tests).
    if args.tests_idxs:
        assert all(0 <= test_idx < len(tests_suit) for test_idx in args.tests_idxs)
        tests_suit = tests_suit.filter_tests_by_idx(args.tests_idxs)

    print('Tests suit:')
    for test in tests_suit:
        print(f'    {test.get_full_name()}')

    # Update the timeout limitations for tests to be proportional
    # to the execution time of the staff solution.
    if args.update_tests_timeout:
        timeout_limit_factor_steps_func = {0.5: 4, 1.2: 3.5, 2: 3, 4: 2.5, 6: 2.3, 14: 2.1, 20: 1.7, 25: 1.6, np.inf: 1.4}
        update_tests_suit_timeout_limit_wrt_staff_solution_time(
            tests_suit=tests_suit, nr_processes=args.nr_processes,
            store_execution_log=args.store_execution_log,
            test_exec_timeout_limit_factor=timeout_limit_factor_steps_func)
    else:
        load_tests_suit_timeout_limit_from_stored_file(tests_suit=tests_suit)

    print()
    print()

    all_submissions = Submission.load_all_submissions(args.submissions_ids)

    if args.sample_submissions or args.sample_submissions is None:
        sample_size = DEFAULT_SUBMISSIONS_SAMPLE_SIZE
        if args.sample_submissions is not None and args.sample_submissions >= 1:
            sample_size = int(args.sample_submissions)
        elif args.sample_submissions is not None and 0 <= args.sample_submissions < 1:
            sample_size = math.ceil(len(all_submissions) * float(args.sample_submissions))
        if len(all_submissions) > sample_size:
            all_submissions = np.random.choice(all_submissions, size=sample_size, replace=False)
    run_tests_for_all_submissions(
        tests_suit, all_submissions, store_execution_log=args.store_execution_log,
        use_processes_pool=not args.no_use_processes_pool,
        nr_processes=args.nr_processes)
