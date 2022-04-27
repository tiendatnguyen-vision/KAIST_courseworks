## CS453 Assignment 2: Coverage Profiler for Python

With this assignment, we will try implementing a basic coverage profiler for Python that can measure statement, branch, and condition coverage. Write a program that accepts another Python script file, as well as its arguments (if any required), and prints out the coverage achieved. The skeleton code, called `pcov.py`, contains a specific format. Consider the following target code, `example2.py`, which you can find under the directry `examples`:

```python
import sys 

x = int(sys.argv[1])
y = int(sys.argv[2])

if x == 10 and y == 5:
    print("Hello")
elif y == 1:
    print("Welcome")
else:
    print("Bye")
```

When you invoke `pcov.py` as follows:

```bash
$ python3 pcov.py -t examples/example2.py 5 5
```

it should print out the following (note that `5 5` at the end is consumed as command line arguments for the target script, `example2.py`):

```bash
=====================================
Program Output
=====================================
Bye
=====================================
Statement Coverage: 75.00% (6 / 8)
Branch Coverage: 50.00% (2 / 4)
Condition Coverage: 50.00% (2 / 6)
=====================================
$
```

The profiler should also support the verbose mode (specified by option `-v`), in which actual predicates and subexpression, as well as their evaluations, should be printed.

```bash
$ python3 pcov.py -v -t examples/example2.py 5 5
=====================================
Program Output
=====================================
Bye
=====================================
Statement Coverage: 75.00% (6 / 8)
Branch Coverage: 50.00% (2 / 4)
Condition Coverage: 33.33% (2 / 6)
=====================================
Covered Branches
Line 6: (x == 10 and y == 5) ==> False
Line 8: (y == 1) ==> False
=====================================
Covered Conditions
Line 6: x == 10 ==> False
Line 8: (y == 1) ==> False
=====================================
$
```

### Scopes

Here are clarificatins about the scope of the coverage.
- `IfExp` (e.g., `a = 10 if b > 3 else 5`) and list comprehension (e.g., `[str(x) for x in l if x > 0]`) create branches.
- Statements are executed by taking a branch. Consequently, statement coverage can be computed by counting what is in the corresponding `body` in `ast.For`, `ast.While`, etc. However, there are other ways of covering statements.
- We will assume that `Try` statements create one branch per `except` handler. For example, the `Try` in `examples/example5.py` creates two branches: one for `IOError` and another for `ArithmeticError`. Note that the body of `Try` itself does not create a branch as it is always executed.
- You need to preserve both short-circuit behaviour and side-effects. For example, consider the verbose example for `python3 pcov.py -v -t examples/example2.py 5 5` above: `y == 5` is not evaluated, therefore no condition coverage is recorded. 

### Skeleton and Test Code

This repository includes a skeleton code named `pcov.py` for your profiler. Please keep the existing code and the command line interface provided, so that GitHub Classroom can run the automated grading scripts. The usage is:

```bash
$ python pcov.py -t [your target python script file] [any command-line arguments]
```

The repository also includes public test cases: please refer to them for mode detail. For example, while we test for the verbose mode output, parentheses do not really matter (i.e., `(y == 1)` and `y == 1` are the same as a representation of a condition).

### Libraries and Python Version

The template repository is configured with Python 3.9. The `ast` module in version 3.9 supports `unparse`: if you use this, you do not need the dependnece on `astor`. So, ideally, no external library is needed.

### Submission Deadline

You need to submit this assignment before **18:00 on 14th of April, 2021.**

