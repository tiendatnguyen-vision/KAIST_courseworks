from solver.code_solver import HierarchicalCodeSolver
import string

def student_solve(test_str):
    stu_solver = HierarchicalCodeSolver()
    return stu_solver.solve(test_str)

def get_exception(code_sample):
    try:
        exec(code_sample, {})
        return None
    except Exception as e:
        return type(e)

def remove_spaces(some_str):
    return ''.join(some_str.split())

def test_code_dd_1():
    test_str1 = '''
a = 0
b = 1
1 / a + 1
''' # ugly I know
    sol_str1 = '''
a = 0
1 / a + 1
'''
    min_str = student_solve(test_str1)
    assert remove_spaces(min_str) == remove_spaces(sol_str1)
    assert get_exception(min_str) == get_exception(test_str1)

def test_code_dd_2():
    test_str2 = '''
a = 0
b = 1
c = 2
d = []
e = d[b]
'''
    sol_str2 = '''
b = 1
d = []
d[b]
'''
    min_str = student_solve(test_str2)
    assert remove_spaces(min_str) == remove_spaces(sol_str2)
    assert get_exception(min_str) == get_exception(test_str2)

def test_code_dd_3():
    test_str3_template = '''
for i in range(%d):
    a = %d/(100-i)
'''
    test_str3 = '\n'.join([test_str3_template % (i, i) for i in range(1, 102)])
    sol_str3 = '''
for i in range(101):
    101 / (100 - i)
'''
    min_str = student_solve(test_str3)
    assert remove_spaces(min_str) == remove_spaces(sol_str3)
    assert get_exception(min_str) == get_exception(test_str3)

def test_code_dd_4():
    test_str4 = '''
my_dict = {
    "a": 1,
    "b": 2,
    "c": 3
}
my_dict["nonexistantkey"]
'''
    sol_str4 = '''
my_dict = {}
my_dict['nonexistantkey']
'''
    min_str = student_solve(test_str4)
    assert remove_spaces(min_str) == remove_spaces(sol_str4)
    assert get_exception(min_str) == get_exception(test_str4)

def test_code_dd_5():
    test_str5 = '''
korean = "환영합니다 여러분!"
english = "Welcome everybody!"
my_list = []
my_list.append(korean)
my_list.append(english)
for e in my_list:
    e.encode('ascii')
'''
    sol_str5 = '''
korean = '환영합니다 여러분!'
'Welcome everybody!'
my_list = []
my_list.append(korean)
for e in my_list:
    e.encode('ascii')
'''
    min_str = student_solve(test_str5)
    assert remove_spaces(min_str) == remove_spaces(sol_str5)
    assert get_exception(min_str) == get_exception(test_str5)

def test_code_dd_6():
    test_str6 = '''
pred = False
if pred or False:
    a = 0
else:
    a = [1]
b = a + 1
'''
    sol_str6 = '''
pred = False
if pred:
    0
else:
    a = []
a + 1
'''
    min_str = student_solve(test_str6)
    assert remove_spaces(min_str) == remove_spaces(sol_str6)
    assert get_exception(min_str) == get_exception(test_str6)

def test_code_dd_7():
    test_str7_template = '''
%s = %d
'''
    test_str7 = ''.join([test_str7_template % (l, i) 
                        for i, l in enumerate(string.ascii_lowercase[1:])])
    test_str7 += '\na'
    sol_str7 = 'a'
    min_str = student_solve(test_str7)
    assert remove_spaces(min_str) == remove_spaces(sol_str7)
    assert get_exception(min_str) == get_exception(test_str7)

def test_code_dd_8():
    test_str8 = '''
import importlib
a = 'string'
b = 'os'
c = 'anyfunnyname'
d = 'math'
modules = [a, b, c, d]
for name in modules:
    importlib.__import__(name)
'''
    sol_str8 = '''import importlib
a = 'string'
b = 'os'
c = 'anyfunnyname'
d = 'math'
modules = [c]
for name in modules:
    importlib.__import__(name)
'''
    min_str = student_solve(test_str8)
    assert remove_spaces(min_str) == remove_spaces(sol_str8)
    assert get_exception(min_str) == get_exception(test_str8)

def test_code_dd_9():
    test_str9 = '''
def bad_max(input_list):
    right_max = bad_max(input_list[1:])
    left_max = input_list[0]
    return max(left_max, right_max)
bad_max([1])
'''
    sol_str9 = '''
def bad_max(input_list):
    bad_max(input_list[:])
bad_max([])
'''
    min_str = student_solve(test_str9)
    assert remove_spaces(min_str) == remove_spaces(sol_str9)
    assert get_exception(min_str) == get_exception(test_str9)

def test_code_dd_10():
    test_str10 = r'''
piece1 = True
piece2 = 1 + 2 + 3
piece3 = 4 - 1 - 2
bad_str = ""
bad_str += "if %s:\n" % piece1
bad_str += "    * %d" % piece2
bad_str += " + %d" % piece3
exec(bad_str)
'''
    sol_str10 = r'''
piece1 = True
piece2 = 1 + 2 + 3
4 - 1 - 2
bad_str = ''
bad_str += 'if %s:\n' % piece1
bad_str += '    * %d' % piece2
exec(bad_str)
'''
    min_str = student_solve(test_str10)
    assert remove_spaces(min_str) == remove_spaces(sol_str10)
    assert issubclass(get_exception(min_str), get_exception(test_str10))
