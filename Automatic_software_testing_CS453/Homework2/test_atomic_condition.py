from test_pcov import *

def test_example6_statement():
    output = run_pcov("examples/example6.py")
    cov, covered, total = get_coverage(STATEMENT_COV, output)
    assert covered == 7
    assert total == 9

def test_example6_branch():
    output = run_pcov("examples/example6.py")
    cov, covered, total = get_coverage(BRANCH_COV, output)
    assert covered == 1
    assert total == 2

def test_example6_condition():
    output = run_pcov("examples/example6.py")
    cov, covered, total = get_coverage(CONDITION_COV, output)
    assert covered == 1
    assert total == 6