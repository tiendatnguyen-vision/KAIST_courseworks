# # some_file.py
import sys
import os

curr_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(curr_dir))
sys.path.insert(1, parent_dir)

from bar import simple
from bar import law

def test_simple_bar1():
    assert simple.bar1() == -7

def test_simple_bar2():
    assert simple.bar2() == "foobar"

def test_simple_bar3():
    assert simple.bar3(2, 3, 1) == -1

def test_simple_bar4():
    assert simple.bar4(2, 3) == True 

def test_simple_bar4_1():
    assert simple.bar4(3, 3) == False 

def test_simple_bar4_2():
    assert simple.bar4(3, 2) == 5 

def test_simple_bar5():
    assert simple.bar5(3) == -4

def test_simple_bar6():
    assert simple.bar6() == -7

def test_simple_bar7():
    assert simple.bar7(3, 1, 4) == 7

def test_simple_bar7_1():
    assert simple.bar7(3, 3, 4) == 0

def test_law_bar1():
    assert law.bar1(1, 2, 2, 1) == 1

def test_law_bar1_1():
    assert law.bar1(2, 0, 9, 9) == 0

def test_law_bar1_2():
    assert law.bar1(1, 1, 21, 46) == 21

def test_law_bar1_3():
    assert law.bar1(1, 0, 3, 4) == 4

def test_law_bar2():
    assert law.bar2(2, 0, 3) == 2

def test_law_bar2_1():
    assert law.bar2(0, 2, 2) == 0

def test_law_bar2_2():
    assert law.bar2(23, 1, 23) == 23

def test_law_bar2_3():
    assert law.bar2(2, 3, 4) == 4

def test_law_bar2_4():
    assert law.bar2(2, 1, 2) == 2

def test_law_bar2_5():
    assert law.bar2(3, 4, 2) == 7
