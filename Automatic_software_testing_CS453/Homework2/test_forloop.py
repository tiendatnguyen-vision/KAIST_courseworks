from test_pcov import *

def test_example3_1_branch():
    output = run_pcov("examples/example3.py", "1")
    cov, covered, total = get_coverage(BRANCH_COV, output)
    assert covered == 3
    assert total == 4

def test_example3_1_condition():
    output = run_pcov("examples/example3.py", "1")
    cov, covered, total = get_coverage(CONDITION_COV, output)
    assert covered == 1
    assert total == 2

def test_example3_2_branch():
    output = run_pcov("examples/example3.py", "2")
    cov, covered, total = get_coverage(BRANCH_COV, output)
    assert covered == 4
    assert total == 4

def test_example3_2_condition():
    output = run_pcov("examples/example3.py", "2")
    cov, covered, total = get_coverage(CONDITION_COV, output)
    assert covered == 2
    assert total == 2



