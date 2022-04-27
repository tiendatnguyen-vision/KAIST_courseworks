import sys

x = int(sys.argv[1])
try:
    print("Hello!")
    if x == 0:
        raise IOError
    elif x == 1:
        raise ArithmeticError
    else:
        pass
except IOError:
    print("Oh no!")
except ArithmeticError:
    print("Arrgh!")
finally:
    print("It's okay")