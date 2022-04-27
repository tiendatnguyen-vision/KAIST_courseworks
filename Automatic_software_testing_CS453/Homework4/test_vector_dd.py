import random
from solver.vector_solver import VectorSolver

class VectorInput():
    def __init__(self, content: list, content_on: list,
                 fault_num: int = 1, given_faults: tuple = tuple()):
        assert len(content) == len(content_on)
        self._content = content
        self._content_on = content_on
        self._seed_fault(fault_num, given_faults)
    
    def show(self, show_all=False):
        '''Prints the input content.
        When show_all is True, will show all content;
        otherwise only shows active components in this input.'''
        if show_all:
            print('Element | Activated')
            for e, b in zip(self._content, self._content_on):
                print(f'{e} | {b}')
        else:
            active_elems = self.active_content()
            for e in active_elems:
                print(e)

    def _seed_fault(self, fault_num: int, given_faults: tuple):
        if len(given_faults) == 0:
            self._faults = tuple(
                random.sample(
                    range(len(self._content)),
                    k = fault_num
                )
            )
        else:
            self._faults = given_faults
    
    def active_content(self):
        return [e for e, b in zip(self._content, self._content_on) if b]
    
    def __len__(self):
        return sum(map(int, self._content_on))

def test_vectors():
    solver = VectorSolver()
    test_num = 100
    failed_tests = []
    wrong_answers = []
    for test_idx in range(test_num):
        test_len = random.randint(10, 100)
        test_input = VectorInput(
            content = range(test_len), 
            content_on = [True]*test_len, 
            fault_num = random.randint(1, 2)
        )
        minimized_test = solver.solve(test_input)
        success = len(minimized_test) == len(test_input._faults)
        success &= all([a == b for a, b in zip(
            minimized_test.active_content(),
            sorted(test_input._faults)
        )])
        if not success:
            failed_tests.append(test_input)
            wrong_answers.append(minimized_test)
    assert len(failed_tests) == 0, f'You failed in input {failed_tests[0].show()}:\nThe faults were {failed_tests[0]._faults} and you gave {wrong_answers[0].show()}'