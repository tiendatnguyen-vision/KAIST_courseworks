'''General tests.

Their main purpose is:
 1) Generate the mutations and kill matrices, used in later tests.
 2) Check that the command line interface is designed as specified.
'''

import subprocess as sp
from test_config import *

def run_pmut_mutate(source, mut_loc):
    cmd = ['python3.9', 'pmut.py', '--action', 'mutate', '--source', source, '--mutants', mut_loc]
    process = sp.run(cmd, capture_output=True, text=True)
    return process.stdout

def run_pmut_execute(source, kill_loc, mut_loc = None):
    cmd = ['python3.9', 'pmut.py', '--action', 'execute', '--source', source, '--kill', kill_loc]
    if mut_loc is not None:
        cmd += ['--mutants', mut_loc]
    print(cmd)
    process = sp.run(cmd, capture_output=True, text=True)
    return process.stdout

def parse_mutate_output(out_lines):
    mut_file_line, mut_gen_line = out_lines
    mut_file_num = int(mut_file_line.split(':')[1].strip())
    mut_num = int(mut_gen_line.split(':')[1].strip())
    return mut_file_num, mut_num

def parse_execute_output(out_lines):
    test_line, kill_line, score_line = out_lines
    test_num = int(test_line.split(':')[1].strip())
    kill_num = int(kill_line.split(':')[1].strip())
    mut_float_score = float(score_line[score_line.index(':')+1:score_line.index('%')])
    mut_killed = int(score_line[score_line.index('(')+1:score_line.index('/')-1])
    mut_total = int(score_line[score_line.index('/')+2:score_line.index(')')])
    return test_num, kill_num, mut_float_score, mut_killed, mut_total

def test_foo_mutate():
    foo_out = run_pmut_mutate('./target/foo/', STU_FOO_MUT_DIR)
    out_lines = foo_out.strip().split('\n')
    assert len(out_lines) == 2
    mut_file_num, mut_num = parse_mutate_output(out_lines)
    assert mut_file_num == 2
    assert mut_num == 93
    
def test_bar_mutate():
    bar_out = run_pmut_mutate('./target/bar/', STU_BAR_MUT_DIR)
    out_lines = bar_out.strip().split('\n')
    assert len(out_lines) == 2
    mut_file_num, mut_num = parse_mutate_output(out_lines)
    assert mut_file_num == 3
    assert mut_num == 182

def test_foo_execute():
    mut_loc = STU_FOO_MUT_DIR if USE_MUT_IN_EXEC else None
    foo_out = run_pmut_execute('./target/foo/', STU_FOO_KM_PATH, mut_loc)
    out_lines = foo_out.strip().split('\n')
    assert len(out_lines) == 3
    t_num, k_num, mut_score, m_killed, m_total = parse_execute_output(out_lines)
    assert t_num == 21
    assert k_num == 79
    assert (mut_score - 84.95) < 1e-1
    assert m_killed == k_num
    assert m_total == 93

def test_bar_execute():
    mut_loc = STU_BAR_MUT_DIR if USE_MUT_IN_EXEC else None
    bar_out = run_pmut_execute('./target/bar/', STU_BAR_KM_PATH, mut_loc)
    out_lines = bar_out.strip().split('\n')
    assert len(out_lines) == 3
    t_num, k_num, mut_score, m_killed, m_total = parse_execute_output(out_lines)
    assert t_num == 23
    assert k_num == 154
    assert (mut_score - 84.62) < 1e-1
    assert m_killed == k_num
    assert m_total == 182