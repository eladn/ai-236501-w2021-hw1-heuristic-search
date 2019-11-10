import os
import sys
import shutil
import time
from typing import *
from functools import partial
from staff_aux.checking_automation.tests_utils import *
import numpy as np
from pprint import pprint

"""

Goes throughout submitted directories. For each submission:
1. find the path the relevant python files.
2. if not consistent with the assignment instruction - mark the `followed-assignment-instruction-test` as failed.
3. open new directory for the tests.
4. copy our files to this new directory.
5. copy the relevant submitted files to this directory.
6. run tests from this dir in dedicated processes.
   use processes from process pool.
   make these processes fail after timeouts.

"""


def submission_tests_invoker(
        submission: Submission, tests_suit: SubmissionTestsSuit, store_execution_log: bool = False) -> Dict[int, float]:
    return submission.run_tests_suit_in_tests_environment(tests_suit, store_execution_log=store_execution_log)


def run_tests_for_all_submissions(
        tests_suit: SubmissionTestsSuit,
        all_submissions: List[Submission],
        use_processes_pool: bool = True,
        store_execution_log: bool = False) -> Dict[Tuple[int], Dict[int, float]]:
    tests_suit_overall_exec_time = tests_suit.calc_overall_tests_execution_time()

    print('Tests suit contains {nr_tests} tests. Total execution time: {low_exec_time:.2f} - {high_exec_time:.2f} min (per submission).'.format(
        nr_tests=len(tests_suit),
        low_exec_time=tests_suit_overall_exec_time / 60,
        high_exec_time=NR_ATTEMPTS_PER_TEST*tests_suit_overall_exec_time / 60
    ))

    print('Running tests on {nr_submissions} submissions.'.format(nr_submissions=len(all_submissions)))

    from multiprocessing import Pool
    process_pool = Pool(NR_PROCESSES)
    jobs_status = {'total_nr_jobs': 0, 'nr_completed_jobs': 0, 'nr_failed_jobs': 0,
                   'start_time': None, 'last_completed_job_time': None}
    failed_submissions = []
    tests_execution_times_per_submission = {}

    def print_jobs_progress():
        nr_finished_jobs = jobs_status['nr_completed_jobs'] + jobs_status['nr_failed_jobs']
        nr_remaining_jobs = jobs_status['total_nr_jobs'] - nr_finished_jobs
        running_time_until_last_completed_job = jobs_status['last_completed_job_time'] - jobs_status['start_time']
        avg_time_per_job = running_time_until_last_completed_job / nr_finished_jobs
        print(
            '{nr_finished}/{tot_nr_jobs} jobs finished. {nr_success} completed successfully. {nr_failed} failed. {nr_remaining_jobs} remain.'.format(
                nr_finished=nr_finished_jobs,
                tot_nr_jobs=jobs_status['total_nr_jobs'],
                nr_success=jobs_status['nr_completed_jobs'],
                nr_failed=jobs_status['nr_failed_jobs'],
                nr_remaining_jobs=nr_remaining_jobs
            ))
        print(
            'Running time until last completed job: {running_time_until_last_completed_job:.2f} min. Avg time per job: {avg_time_per_job:.2f} min. Estimated remaining time: {est_remaining_time:.2f} min.'.format(
                running_time_until_last_completed_job=running_time_until_last_completed_job / 60,
                avg_time_per_job=avg_time_per_job / 60,
                est_remaining_time=avg_time_per_job * nr_remaining_jobs / 60
            ))

    def submission_tests_invoker__on_success(submission: Submission, returned_value):
        jobs_status['last_completed_job_time'] = time.time()
        jobs_status['nr_completed_jobs'] += 1
        print('========   Successfully completed running tests for submission: ids={ids}   ========'.format(ids=submission.ids))
        print_jobs_progress()
        print()
        tests_execution_times_per_submission[tuple(submission.ids)] = returned_value

    def submission_tests_invoker__on_error(submission: Submission, error):
        print()
        print('XXXXXXXX   FAILED running tests for submission: ids={ids}   XXXXXXXX'.format(ids=submission.ids))
        print(error)
        print()
        failed_submissions.append(submission)
        jobs_status['last_completed_job_time'] = time.time()
        jobs_status['nr_failed_jobs'] += 1
        print_jobs_progress()
        with open(os.path.join(submission.test_logs_dir_path, 'test-run-stderr.txt'), 'w') as test_run_stderr:
            test_run_stderr.write(str(error))

    jobs_status['start_time'] = time.time()
    for submission in all_submissions:
        if use_processes_pool:
            print('Spawning tests worker for submission: ids: {ids} -- submission-dir: {submission_dir}'.format(
                ids=submission.ids,
                submission_dir=submission.main_submission_directory_name
            ))
            jobs_status['total_nr_jobs'] += 1
            process_pool.apply_async(
                submission_tests_invoker, (submission, tests_suit, store_execution_log),
                callback=partial(submission_tests_invoker__on_success, submission),
                error_callback=partial(submission_tests_invoker__on_error, submission))
        else:
            print('Running tests for submission: ids: {ids} -- submission-dir: {submission_dir}'.format(
                ids=submission.ids,
                submission_dir=submission.main_submission_directory_name
            ))
            submission_tests_invoker(submission, tests_suit, store_execution_log)
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
        test_exec_timeout_limit_factor: Union[float, Dict[float, float]] = 2):

    nr_staff_solution_executions = NR_PROCESSES * 3

    print('Running the tests suit over the staff solution for {executions} times in order to calculate the execution timeout limit for each test.'.format(
        executions=nr_staff_solution_executions
    ))
    print('The timeout calculation for each test takes into account the maximum time between these executions.')

    new_staff_solution_submissions = [
        copy_staff_solution_as_submission(int(str(STAFF_SOLUTION_DUMMY_ID) + str(execution_idx)))
        for execution_idx in range(nr_staff_solution_executions)
    ]
    results_for_all_staff_submissions = run_tests_for_all_submissions(
        tests_suit, new_staff_solution_submissions, store_execution_log=store_execution_log)
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
        return factored_sum

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

    with open(os.path.join(TESTS_LOGS_PATH, 'tests-calculated-timeouts.txt'), 'w') as calculated_timeouts_file:
        for test in tests_suit:
            calculated_timeouts_file.write(test.get_name())
            calculated_timeouts_file.write('\n')
            calculated_timeouts_file.write('execution timeout limit: {timeout:.4f}\n\n'.format(
                timeout=new_time_per_test[test.index]))

    compared_times_per_test = {
        test.index: {
            'orig': test.execution_timeout,
            'staff': staff_solution_time_per_test[test.index],
            'new': new_time_per_test[test.index]}
        for test in tests_suit
    }

    print('Compared new-vs-old tests execution timeout limitation: ')
    pprint({tests_suit.get_test_by_idx(test_idx).get_name(): times
            for test_idx, times in compared_times_per_test.items()})
    print()

    tests_suit.update_timeouts(new_time_per_test)

    for staff_submission in new_staff_solution_submissions:
        staff_submission.remove_all_files()


def copy_staff_solution_as_submission(submission_id: int) -> Submission:
    staff_solution_submission_path = os.path.join(SUBMISSIONS_PATH, str(submission_id))
    if os.path.isdir(staff_solution_submission_path):
        shutil.rmtree(staff_solution_submission_path)
    shutil.copytree(STAFF_SOLUTION_CODE_PATH, staff_solution_submission_path)
    return Submission(staff_solution_submission_path)


if __name__ == '__main__':
    tests_suit = DeliveriesTestsSuitCreator.create_tests_suit()

    tests_suit = tests_suit.filter_tests_by_idx([1, 13])

    # [optional] 1st arg may contain a list of test indeces to run (otherwise run all tests).
    only_tests_idxs = None
    if len(sys.argv) >= 2:
        only_tests_idxs = [int(test_idx) for test_idx in sys.argv[1].split(',')]  # TODO: handle invalid input format
        assert all(0 <= test_idx < len(tests_suit) for test_idx in only_tests_idxs)
        tests_suit = tests_suit.filter_tests_by_idx(only_tests_idxs)

    # [optional] 2nd arg may indicate whether to store the execution log.
    # TODO: consider having a list of test indeces to store exec log for.
    store_execution_log = bool(sys.argv[2]) if len(sys.argv) >= 3 else False

    # Update the timeout limitations for tests to be proportional
    # to the execution time of the staff solution.
    timeout_limit_factor_steps_func = {0.5: 4, 1.2: 3.5, 2: 3, 4: 2.5, 6: 1.7, 14: 1.6, 20: 1.55, 25: 1.5, np.inf: 1.4}
    update_tests_suit_timeout_limit_wrt_staff_solution_time(
        tests_suit,
        store_execution_log=store_execution_log,
        test_exec_timeout_limit_factor=timeout_limit_factor_steps_func)
    print()
    print()

    all_submissions = Submission.load_all_submissions()
    run_tests_for_all_submissions(tests_suit, all_submissions, store_execution_log=store_execution_log)
