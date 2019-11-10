import os
import shutil


PDFS_PATH = '/Users/eladn/Documents/ai-w2019-hw1/feedback/'
TESTS_RES_PATH = '/Users/eladn/Documents/ai-w2019-hw1-submissions/tests-logs-v6'

tests_res_dirs = set(d for d in os.listdir(TESTS_RES_PATH) if os.path.isdir(os.path.join(TESTS_RES_PATH, d)))
pdfs_dirs = set(d for d in os.listdir(PDFS_PATH) if os.path.isdir(os.path.join(PDFS_PATH, d)))

print(pdfs_dirs.symmetric_difference(tests_res_dirs))

pdfs_dirs_paths = {os.path.join(PDFS_PATH, d) for d in pdfs_dirs}
tests_res_dirs_paths = {os.path.join(TESTS_RES_PATH, d) for d in tests_res_dirs}

assert all(len(os.listdir(dp)) == 1 and os.listdir(dp)[0].split('.')[-1] == 'pdf' for dp in pdfs_dirs_paths)

for submission_dir in pdfs_dirs:
    pdf_dirpath = os.path.join(PDFS_PATH, submission_dir)
    assert os.path.isdir(pdf_dirpath)
    tests_res_dirpath = os.path.join(TESTS_RES_PATH, submission_dir)
    assert os.path.isdir(tests_res_dirpath)
    assert len(os.listdir(pdf_dirpath)) == 1
    pdf_filename = os.listdir(pdf_dirpath)[0]
    assert pdf_filename.split('.')[-1] == 'pdf'
    pdf_filepath = os.path.join(pdf_dirpath, pdf_filename)
    assert os.path.isfile(pdf_filepath)
    shutil.copy2(pdf_filepath, tests_res_dirpath)
    
