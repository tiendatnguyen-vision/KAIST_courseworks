from test_pcov import *

def test_example2_statement():
    output = run_pcov("examples/example2.py", "10", "1")
    cov, covered, total = get_coverage(STATEMENT_COV, output)
    assert covered == 6
    assert total == 8

def test_example2_branch():
    output = run_pcov("examples/example2.py", "10", "1")
    cov, covered, total = get_coverage(BRANCH_COV, output)
    assert covered == 2
    assert total == 4

def test_example2_condition():
    output = run_pcov("examples/example2.py", "10", "1")
    cov, covered, total = get_coverage(CONDITION_COV, output)
    assert covered == 3
    assert total == 6

def test_example2_verbose():
    output = run_pcov_verbose("examples/example2.py", "10", "1")
    lines = get_verbose_output(output)
    tuples = process_verbose_output(lines)
    
    assert (6, False) == tuples[0]
    assert (8, True) == tuples[1]
    assert (6, True) == tuples[2]
    assert (6, False) == tuples[3]
    
    assert lines[0].find("x == 10 and y == 5") > 0
    assert lines[1].find("y == 1") > 0
    assert lines[2].find("x == 10") > 0
    assert lines[3].find("y == 5") > 0