from test_pcov import *

def test_example5_statement():
    output = run_pcov("examples/example5.py", "1")
    cov, covered, total = get_coverage(STATEMENT_COV, output)
    assert covered == 9
    assert total == 12

def test_example5_branch():
    output = run_pcov("examples/example5.py", "1")
    cov, covered, total = get_coverage(BRANCH_COV, output)
    assert covered == 3
    assert total == 6

def test_example5_condition():
    output = run_pcov("examples/example5.py", "1")
    cov, covered, total = get_coverage(CONDITION_COV, output)
    assert covered == 2
    assert total == 4

def test_example5_except_statement():
    output = run_pcov("examples/example5.py", "0")
    cov, covered, total = get_coverage(STATEMENT_COV, output)
    assert covered == 8
    assert total == 12

def test_example5_except_branch():
    output = run_pcov("examples/example5.py", "0")
    cov, covered, total = get_coverage(BRANCH_COV, output)
    assert covered == 2
    assert total == 6

def test_example5_except_condition():
    output = run_pcov("examples/example5.py", "0")
    cov, covered, total = get_coverage(CONDITION_COV, output)
    assert covered == 1
    assert total == 4


def test_example5_verbose():
    output = run_pcov_verbose("examples/example5.py", "0")
    lines = get_verbose_output(output)
        
    assert lines[0].find("try ==> IOError") > 0
    assert lines[1].find("x == 0") > 0
    assert lines[2].find("x == 0") > 0