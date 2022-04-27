from solver.solver import Solver

class VectorEvaluator():
    def evaluate(self, v):
        indicate = 1
        active_content = v.active_content()
        for ele in v._faults:
            if ele not in active_content:
                indicate-=1
        if indicate == 1:
            return True
        else:
            return False

def copy_and_set_vector(vector, true_positions):
    import copy
    vector1 = copy.deepcopy(vector)
    for i in range(len(vector1._content)):
        if i not in true_positions:
            vector1._content_on[i] = False
        else:
            vector1._content_on[i] = True
    return vector1

def DD(vector, keep_list):
    import copy
    vector0 = copy.deepcopy(vector)
    if len(keep_list) == 1:
        return keep_list
    n = len(keep_list)
    fixed_area = [i for i in vector0._content if i not in keep_list]
    area1 = keep_list[:n//2]
    area2 = keep_list[n//2:]
    P1 = fixed_area + area1
    P2 = fixed_area + area2
    vector1 = copy_and_set_vector(vector0, P1)
    vector2 = copy_and_set_vector(vector0, P2)
    if VectorEvaluator().evaluate(vector1) == True:
        return DD(vector1, area1)
    elif VectorEvaluator().evaluate(vector2) == True:
        return DD(vector2, area2)
    else:
        return DD(vector2, area1) + DD(vector1, area2)


class VectorSolver(Solver):
    def __init__(self):
        pass
    def solve(self, input_vector):
        original_area = input_vector.active_content()
        minimized_area = DD(input_vector, original_area)
        input_vector = copy_and_set_vector(input_vector, minimized_area)
        return input_vector






