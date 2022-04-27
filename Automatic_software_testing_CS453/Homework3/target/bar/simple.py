def bar1():
    a = 3
    b = 4
    c = a + b
    return -c

def bar2():
    a = 'foo'
    b = 'bar'
    return a + b

def bar3(x, y, c):
    y = -1 * y
    z = (x + y) * c
    return z

def bar4(a, b):
    if a > b:
        return a + b
    elif a == b:
        return False
    else:
        return True

def bar5(a):
    a += 1
    return bar3(3, 4, a)

def bar6():
    a = bar1()
    return a

def bar7(a, b, c):
    if a > b:
        return a | c
    else:
        return a & c