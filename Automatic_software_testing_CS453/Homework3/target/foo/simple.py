def foo1(a, b):
    if a < b:
        return a + b
    else:
        return a - b

def foo2(a, b):
    if a < b:
        return a * b
    elif a == b:
        return b * 3
    else:
        return -2

def foo3(a, b):
    if a >= b:
        return foo1(a, b)
    else:
        return foo2(a, b)

def foo4(a):
    x = -a
    return str(34) + ' + ' + str(x)