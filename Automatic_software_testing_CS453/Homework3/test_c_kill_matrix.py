'''Execute tests.

These tests evaluate whether the generated kill matrix is correct. 
The kill matrix is compared with a reference solution.
'''

import os
import json
import numpy as np
from test_config import *

def match_dict_idx(ref_dict, stu_dict):
    ref_inv = {idx:name for name, idx in ref_dict.items()}
    ref2stu_idx = [stu_dict[ref_inv[i]] for i in range(len(ref_dict))]
    return ref2stu_idx

def test_foo_km():
    stu_path = STU_FOO_KM_PATH
    ref_test_idx = json.load(open(os.path.join(REF_FOO_KM_PATH, TEST_JSON_FILE)))
    stu_test_idx = json.load(open(os.path.join(stu_path, TEST_JSON_FILE)))
    row_permutation = match_dict_idx(ref_test_idx, stu_test_idx)
    ref_mut_idx = json.load(open(os.path.join(REF_FOO_KM_PATH, MUT_JSON_FILE)))
    stu_mut_idx = json.load(open(os.path.join(stu_path, MUT_JSON_FILE)))
    col_permutation = match_dict_idx(ref_mut_idx, stu_mut_idx)
    
    ref_matrix = np.loadtxt(os.path.join(REF_FOO_KM_PATH, MATRIX_FILE), dtype=np.int32)
    stu_matrix = np.loadtxt(os.path.join(stu_path, MATRIX_FILE), dtype=np.int32)
    stu_shift_matrix = stu_matrix[row_permutation, :]
    stu_shift_matrix = stu_shift_matrix[:, col_permutation]
    assert np.all(ref_matrix == stu_shift_matrix)

def test_bar_km():
    stu_path = STU_BAR_KM_PATH
    ref_test_idx = json.load(open(os.path.join(REF_BAR_KM_PATH, TEST_JSON_FILE)))
    stu_test_idx = json.load(open(os.path.join(stu_path, TEST_JSON_FILE)))
    row_permutation = match_dict_idx(ref_test_idx, stu_test_idx)
    ref_mut_idx = json.load(open(os.path.join(REF_BAR_KM_PATH, MUT_JSON_FILE)))
    stu_mut_idx = json.load(open(os.path.join(stu_path, MUT_JSON_FILE)))
    col_permutation = match_dict_idx(ref_mut_idx, stu_mut_idx)
    
    ref_matrix = np.loadtxt(os.path.join(REF_BAR_KM_PATH, MATRIX_FILE), dtype=np.int32)
    stu_matrix = np.loadtxt(os.path.join(stu_path, MATRIX_FILE), dtype=np.int32)
    stu_shift_matrix = stu_matrix[row_permutation, :]
    stu_shift_matrix = stu_shift_matrix[:, col_permutation]
    assert np.all(ref_matrix == stu_shift_matrix)

