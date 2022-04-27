from test_pcov import *

def test_example4_statement():
    output = run_pcov("examples/example4.py", "2", "1", "2", "3")
    cov, covered, total = get_coverage(STATEMENT_COV, output)
    assert covered == 6
    assert total == 6

def test_example4_branch():
    output = run_pcov("examples/example4.py", "2", "1", "2", "3")
    cov, covered, total = get_coverage(BRANCH_COV, output)
    assert covered == 3
    assert total == 4

def test_example4_condition():
    output = run_pcov("examples/example4.py", "10", "1", "2", "3")
    cov, covered, total = get_coverage(CONDITION_COV, output)
    assert covered == 2
    assert total == 4

def test_example4_verbose():
    output = run_pcov_verbose("examples/example4.py", "10", "1", "2", "3")
    lines = get_verbose_output(output)
    tuples = process_verbose_output(lines)
    
    assert (4, False) == tuples[0]
    assert (7, False) == tuples[1]
    assert (4, False) == tuples[2]
    assert (7, False) == tuples[3]
    
    assert lines[0].find("int(x) > threshold") > 0
    assert lines[1].find("count_gt(int(sys.argv[1]), sys.argv[2:]) > 0") > 0
    assert lines[2].find("int(x) > threshold") > 0
    assert lines[3].find("count_gt(int(sys.argv[1]), sys.argv[2:]) > 0") > 0