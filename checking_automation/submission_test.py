import os
import sys
import time
import numpy as np
from typing import *

# if os.path.basename(os.getcwd().rstrip('/')) == 'ai-236501-w2019-hw1':
#     new_path = os.path.join(os.getcwd(), 'staff_aux/staff_solution/')
#     if os.getcwd() in sys.path:
#         sys.path.remove(os.getcwd())
#     sys.path.append(new_path)
#     os.chdir(new_path)
# print(os.getcwd())

from framework import *
from deliveries import *
from tests_utils import *

# Load the streets map
streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path("tlv_streets_map.csv"))

# Make `np.random` behave deterministic.
# Notice: It is set again on `test.run_test()`, using the test index to also effect the seed.
np.random.seed(NUMPY_RANDOM_SEED_BASE)


def run_tests_and_write_results():
    assert len(sys.argv) >= 2
    results_output_file_path = sys.argv[1]
    single_test_idx = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    execution_log_file_path = str(sys.argv[3]) if len(sys.argv) >= 4 else False
    tests_suit = DeliveriesTestsSuitCreator.create_tests_suit()
    if single_test_idx is not None:
        assert 0 <= single_test_idx < len(tests_suit)
        tests_suit = tests_suit.filter_tests_by_idx([single_test_idx])
    if bool(execution_log_file_path):
        execution_log_file = open(execution_log_file_path, 'w')
    with open(results_output_file_path, 'w') as results_output_file:
        for test in tests_suit:
            test_result, res, execution_log = test.run_test(streets_map, store_execution_log=bool(execution_log_file_path))
            print(res)
            assert test_result is not None
            results_output_file.write(test_result.serialize())
            results_output_file.write('\n')
            if bool(execution_log_file_path):
                import json
                execution_log_dump = json.dumps(execution_log, sort_keys=True, indent=4)
                execution_log_file.write(execution_log_dump)
                execution_log_file.write('\n\n')
    if bool(execution_log_file_path):
        execution_log_file.close()


if __name__ == '__main__':
    run_tests_and_write_results()
