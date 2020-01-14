import os
import shutil
from typing import Dict


PDFS_PATH = '/Users/eladn/Documents/ai_HW1_dry_feedback/'
TESTS_RES_PATH = '/Users/eladn/Documents/ai-w2020-hw1-submissions-checking/tests-logs'

tests_res_dirs = set(d for d in os.listdir(TESTS_RES_PATH) if os.path.isdir(os.path.join(TESTS_RES_PATH, d)))
pdfs_dirs = set(d for d in os.listdir(PDFS_PATH) if os.path.isdir(os.path.join(PDFS_PATH, d)))

# TODO: check all submission ids in both folders (pdfs dir & tests results dir) and verify there is exactly one
#  occurrence for each id.

dir_names_symmetric_difference = pdfs_dirs.symmetric_difference(tests_res_dirs)
if len(dir_names_symmetric_difference):
    print(f'There are {len(dir_names_symmetric_difference)} submission directories that exist in only one place'
          f'(tests results dir / pdfs dir). And here they are: ')
    for dir_name in dir_names_symmetric_difference:
        print(f'    {dir_name} -- in pdfs: {dir_name in pdfs_dirs} -- in tests results: {dir_name in tests_res_dirs}')
        ids = dir_name.split('-')
        for id in ids:
            pdf_dirs_with_this_id = [dirname for dirname in pdfs_dirs if id in dirname]
            if pdf_dirs_with_this_id:
                print(f'        pdf_dirs_with_this_id={pdf_dirs_with_this_id}')
            tests_res_dirs_with_this_id = [dirname for dirname in pdfs_dirs if id in dirname]
            if tests_res_dirs_with_this_id:
                print(f'        tests_res_dirs_with_this_id={tests_res_dirs_with_this_id}')
    print()
else:
    print('All submissions in tests results dir and pdfs dir are exactly the same (ids).')

pdfs_dirs_paths = {os.path.join(PDFS_PATH, d) for d in pdfs_dirs}
tests_res_dirs_paths = {os.path.join(TESTS_RES_PATH, d) for d in tests_res_dirs}

submissions_without_pdf_file_as_expected = [
    os.path.basename(dp) for dp in pdfs_dirs_paths
    if len(os.listdir(dp)) != 1 or os.listdir(dp)[0].split('.')[-1] != 'pdf']

pdf_file_path_per_submission: Dict[str, str] = {}
for pdf_dir_path in pdfs_dirs_paths:
    assert os.path.isdir(pdf_dir_path)
    pdf_dir_name = os.path.basename(pdf_dir_path)
    pdf_files_in_pdf_dir = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(pdf_dir_path) for filename in filenames
        if filename.split('.')[-1] == 'pdf']
    if len(pdf_files_in_pdf_dir) < 1:
        print(f'Submission pdf dir {pdf_dir_name} has no pdf files in it!')
        continue
    if len(pdf_files_in_pdf_dir) > 1:
        print(f'Submission pdf dir {pdf_dir_name} has more than one pdf file in it: {pdf_files_in_pdf_dir}; taking the 1st')
    pdf_file_path_per_submission[pdf_dir_name] = pdf_files_in_pdf_dir[0]

for submission_dir in pdfs_dirs:
    assert submission_dir in pdf_file_path_per_submission
    pdf_filepath = pdf_file_path_per_submission[submission_dir]

    tests_res_dirpath = os.path.join(TESTS_RES_PATH, submission_dir)
    assert os.path.isdir(tests_res_dirpath)

    assert os.path.isfile(pdf_filepath)
    shutil.copy2(pdf_filepath, tests_res_dirpath)
