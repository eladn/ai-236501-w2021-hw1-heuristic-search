import os
import sys
from itertools import zip_longest
from collections import Counter
from staff_aux.checking_automation.tests_utils import *
import numpy as np
from typing import *
from warnings import warn


ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])


def generate_tests_results():
    all_submissions = Submission.load_all_submissions()
    tests_suit = DeliveriesTestsSuitCreator.create_tests_suit()

    print('#submissions: {} -- #tests_pre_submission: {}'.format(len(all_submissions), len(tests_suit)))

    staff_solution_submission = [submission for submission in all_submissions if submission.ids[0] == STAFF_SOLUTION_DUMMY_ID][0]  # TODO: move this logic to `SubmissionsContainer`
    staff_solution_tests_results = staff_solution_submission.load_tests_results(tests_suit)
    assert all(result and isinstance(result, TestResult) for result in staff_solution_tests_results)

    results_matrix = np.zeros((len(all_submissions), len(tests_suit)))
    results_per_submission = []

    for submission_idx, submission in enumerate(all_submissions):
        submission_tests_results = submission.load_tests_results(tests_suit)
        submission_tests_results.calc_pass_vector(staff_solution_tests_results)
        results_per_submission.append((submission_idx, submission, submission_tests_results))
        results_matrix[submission_idx, :] = submission_tests_results.pass_vector

    avg_result_vector = np.average(results_matrix, axis=0)
    print('avg_result_vector:')
    print(avg_result_vector)
    print()

    # NOTICE: The averages include the staff solution submission.
    print('average (tests are equally weighted): {}'.format(np.average(avg_result_vector) * 100))
    print()

    tests_grade_weights_vector = np.ones(len(tests_suit)) * (1/len(tests_suit))
    tests_grade_weights_vector[0:5] *= 0.5
    tests_grade_weights_vector[len(tests_suit) - 5 : len(tests_suit)] *= 1.5
    print('weights vector: ', tests_grade_weights_vector)
    assert abs(np.sum(tests_grade_weights_vector) - 1.0) < 0.000001
    grades = np.round(np.matmul(results_matrix, tests_grade_weights_vector) * 100)
    assert len(grades) == len(all_submissions)

    # NOTICE: The averages include the staff solution submission.
    print('average (weighted): {}'.format(np.average(grades)))
    print()

    nr_100p_correct = sum(1 for _, _, tests_results in results_per_submission if np.all(tests_results.pass_vector == 1))
    print('nr_100p_correct: {}'.format(nr_100p_correct))
    print()

    for submission_idx, submission, submission_tests_results in results_per_submission:
        with open(os.path.join(submission.test_logs_dir_path, 'tests_results_summary.txt'), 'w') as tests_results_file:
            tests_results_file.write('followed-assignment-instructions')
            tests_results_file.write('\n')
            tests_results_file.write('PASS' if submission.followed_assignment_instructions else 'FAIL')
            tests_results_file.write('\n')
            tests_results_file.write('\n')
            for test, test_binary_result in zip(tests_suit, submission_tests_results.pass_vector):
                tests_results_file.write(test.get_full_name())
                tests_results_file.write('\n')
                tests_results_file.write('PASS' if test_binary_result else 'FAIL')
                tests_results_file.write('\n')
                tests_results_file.write('\n')

    with open(os.path.join(TESTS_LOGS_PATH, 'all_submissions_tests_results.csv'), 'w') as tests_results_file:
        tests_results_file.write('id, ')
        tests_results_file.write('followed-assignment-instructions, ')
        tests_results_file.write(', '.join(tests_suit.get_tests_names()))
        tests_results_file.write(', wet-grade')
        tests_results_file.write('\n')
        for submission_idx, submission, submission_tests_results in results_per_submission:
            for stud_id in submission.ids:
                if stud_id == STAFF_SOLUTION_DUMMY_ID:
                    continue
                grade = grades[submission_idx]
                tests_results_file.write(str(stud_id) + ', ')
                tests_results_file.write(str(int(submission.followed_assignment_instructions)) + ', ')
                tests_results_file.write(', '.join(str(val) for val in submission_tests_results.pass_vector))
                tests_results_file.write(', {grade}'.format(grade=grade))
                tests_results_file.write(' \n')

    with open(os.path.join(TESTS_LOGS_PATH, 'tests.txt'), 'w') as tests_numpy_seeds_file:
        for test in tests_suit:
            tests_numpy_seeds_file.write('Test name and index: {}\n'.format(test.get_full_name()))
            tests_numpy_seeds_file.write('Wet grade weight: {weight:.4f}\n'.format(weight=tests_grade_weights_vector[test.index]))
            tests_numpy_seeds_file.write('Numpy seed: {seed}\n'.format(seed=test.calc_seed()))
            tests_numpy_seeds_file.write('\n')

    grade_per_submission = {
        submission: float(np.average(results_matrix[submission_idx:]))
        for submission_idx, submission in enumerate(all_submissions)
    }

    results_distr_per_test = [
        Counter(all_tests_results_for_submission[test.index].path if all_tests_results_for_submission[test.index] is not None else None
                for _, _, all_tests_results_for_submission in results_per_submission)
        for test in tests_suit
    ]

    for test, test_distr in zip(tests_suit, results_distr_per_test):
        print('Test: {}'.format(test.get_full_name()))
        print([(freq,) + ('staff' if solution == staff_solution_tests_results[test.index].path else ('TIMEOUT' if solution is None else 'other'),)
               for solution, freq in test_distr.most_common()])
        print()

    sys.stdout.flush()
    sys.stderr.flush()

    for test, test_distr in zip(tests_suit, results_distr_per_test):
        solutions_with_freq_ordered_by_freq = test_distr.most_common()
        if len(solutions_with_freq_ordered_by_freq) < 2:
            continue

        all_freqs = list(set(freq for _, freq in solutions_with_freq_ordered_by_freq))
        all_freqs.sort(reverse=True)
        freq_to_commonness_order_mapping = {freq: k for k, freq in enumerate(all_freqs, start=1)}

        staff_solution = staff_solution_tests_results[test.index].path
        staff_solution_freq = [freq for sol, freq in solutions_with_freq_ordered_by_freq if sol == staff_solution][0]

        most_common_solution_freq = solutions_with_freq_ordered_by_freq[0][1]
        most_common_solutions = [sol for sol, freq in solutions_with_freq_ordered_by_freq
                                 if freq == most_common_solution_freq]
        assert len(most_common_solutions) >= 1

        # Sanity check: the staff solution is one of the most common results.
        if staff_solution not in most_common_solutions:
            warn('\nTest: {test_name}. \n\tThe staff solution is not one of the most common solutions. \n\tStaff solution freq: {staff_solution_freq} ({kth} most common). \n\tMost common solution freq: {most_common_solution_freq}. \n\t'.format(
                test_name=test.get_full_name(),
                staff_solution_freq=staff_solution_freq,
                kth=ordinal(freq_to_commonness_order_mapping[staff_solution_freq]),
                most_common_solution_freq=most_common_solution_freq))

        # Sanity check: there is only one most common solution.
        if len(most_common_solutions) > 1:
            warn('\nTest: {test_name}. \n\tThere are multiple solutions with the highest frequency ({freq}). \n\t'.format(
                test_name=test.get_full_name(), freq=most_common_solution_freq))

        for sol, freq in solutions_with_freq_ordered_by_freq:
            if sol == staff_solution_tests_results[test.index].path:
                continue

            if staff_solution_freq > freq * 5:
                continue

            submissions_used_this_solution = [
                submission for _, submission, all_tests_results_for_submission in results_per_submission
                if (all_tests_results_for_submission[test.index] is None and sol is None) or \
                   (all_tests_results_for_submission[test.index] is not None and \
                   all_tests_results_for_submission[test.index].path == sol)
            ]
            highest_total_grade_within_submissions_used_this_solution = max(grade_per_submission[submission] for submission in submissions_used_this_solution)
            submissions_used_this_solution_whos_total_grade_is_best = [
                submission
                for submission in submissions_used_this_solution
                if abs(grade_per_submission[submission] - highest_total_grade_within_submissions_used_this_solution) < 0.001
            ]
            warn('\nTest: {test_name}. \n\tFound non-acceptable solution with freq {freq} ({kth} most common), while staff solution freq is {staff_solution_freq}. \n\tThe staff solution is:        {staff_solution}. \n\tThe non-accepted solution is: {nonaccepted_solution}. \n\tSubmissions who produced it: {submissions_used_this_solution}. \n\tHighest total grade within submissions used this solution: {highest_total_grade_within_submissions_used_this_solution}. \n\tSubmissions who produced it and have the highest total grade: {submissions_used_this_solution_whos_total_grade_is_best}. You might want to check these manually.\n\t'.format(
                freq=freq,
                kth=ordinal(freq_to_commonness_order_mapping[freq]),
                staff_solution_freq=staff_solution_freq,
                test_name=test.get_full_name(),
                staff_solution=staff_solution,
                nonaccepted_solution=sol if sol is not None else 'TIMEOUT',
                submissions_used_this_solution=[submission.ids for submission in submissions_used_this_solution],
                highest_total_grade_within_submissions_used_this_solution=highest_total_grade_within_submissions_used_this_solution,
                submissions_used_this_solution_whos_total_grade_is_best=[submission.ids for submission in submissions_used_this_solution_whos_total_grade_is_best]))


if __name__ == '__main__':
    generate_tests_results()
