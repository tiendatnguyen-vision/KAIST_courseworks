print("Here we go")
x = [(i, j) for i in range(10) if i % 2 == 0 for j in range(i) if j > 4]
print(x)
print(x)
print(x)
print(x)
print(x)
print(x)
print(x)
x = [(b, a) for b in range(10) if b % 2 == 0 for a in range(b) if a > 4]