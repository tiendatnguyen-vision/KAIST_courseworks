# CS453 Assignment 4: Delta Debugging

Welcome to your last assignment. Here, we implement the **Delta Debugging** (DD) algorithm described in the [April 26th](https://coinse.kaist.ac.kr/assets/files/teaching/cs453/cs453slide08.pdf) lecture.

For those interested, you can find related papers below:
 - [The original delta debugging introduction](http://web2.cs.columbia.edu/~junfeng/09fa-e6998/papers/delta-debug.pdf) (back from '99)
 - [Hierachical delta debugging](https://dl.acm.org/doi/10.1145/1134285.1134307) (introduced in ICSE'06)
 - [Instead of `ddmin`, how about `ddmax`?](https://publications.cispa.saarland/3060/1/camera-ready-submission.pdf) (recent work from ICSE'20)

## Overview

Given a string code snippet, we want to reduce it to the minimal components that cause the specified error. For example, consider the Python code below.

```python
a = 1
1/0
b = a + 1
```

Executing this code would give a `ZeroDivisionError`. Is all of this code responsible for this exception? Clearly not - only the `1/0` part is. Hence your task: reduce the previous cluttered code to reveal what exactly is responsible for the exception. As a stepping stone, you will also be asked to perform an abstracted version of DD on vectors, similarly to the content from the lecture slides.

## Skeleton and Test Code

Write your code in the `solver` directory. In the `solver/vector_solver.py` file, write your solution for identifying faulty elements in the vector in the `solve` method of the `VectorSolver` class, which should use linear DD. The `VectorSolver.solve` takes a `VectorInput` (defined in `test_code_dd.py`) and returns a minimized input. The test file probably describes the exact specs more concretely than I do, so go take a look.

Similarly, in the `solver/code_solver.py` file, implement the hierarchical version of DD (HDD) in `HierarchicalCodeSolver`. (Using linear DD for the code tests will not only take a long time, but also yield different results from HDD.) Here, the `solve` method is expected to take Python code as a string, and return minimized code as a string. The provided string will be executed to see if the error type is preserved; further, the result of your minimization will be compared with a reference solution.

Specifics:
 - Don't worry about spaces: the tests will ignore whitespace in your return values. 
 - When dealing with odd numbers, split to halves so that the middle element is included in the right-hand side of the split (that is, you can use indices like `[:len(l)//2]`.)
 - You might find that for certain inputs, DD will return nonsensical reductions / non-minimal reductions. _This is normal_. The tests, on the other hand, have been crafted so that well-implemented DD will return sensical output (although it may not be minimal), so if you see ill-formed code as a result of executing your solver on the tests, something is likely wrong.

## Libraries and Python Version
As always, the template repository is configured with Python 3.9; there are no external libraries used.

## Submission Deadline
Submit your solver code by Monday 2021-06-21, 18:00.
