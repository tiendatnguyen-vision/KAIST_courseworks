import ast
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mutation Testing Tool.')
    parser.add_argument('-a', '--action', choices=["mutate", "execute"], required=True)
    parser.add_argument('-s', '--source', type=str, required=True)
    parser.add_argument('-m', '--mutants', type=str, required=False)
    parser.add_argument('-k', '--kill', type=str)

    args = parser.parse_args()
    if args.action == "execute" and not args.kill:
        parser.error("Mutant execution action requires -k/--kill")

    #########################################
    # From here
    import os
    import copy
    import astor
    sys.dont_write_bytecode = True
    # python3 pmut.py --action mutate --source target/bar --mutants nhap/Mutants

    class Visit_mutate_tree(ast.NodeVisitor):
        def __init__(self, dict_mutation_):
            self.dict_mutation_ = dict_mutation_

        def generic_visit(self, node):
            try:
                lineno_, col_offset_, end_lineno_, end_col_offset_ = node.lineno, node.col_offset, node.end_lineno, node.end_col_offset
            except:
                ast.NodeVisitor.generic_visit(self, node)
            else:
                if isinstance(node, ast.Compare) or isinstance(node, ast.AugAssign) or isinstance(node,
                                                                                                  ast.UnaryOp) or isinstance(
                        node, ast.BinOp) or isinstance(node, ast.Return) or isinstance(node, ast.Constant):
                    if (lineno_, col_offset_, end_lineno_, end_col_offset_) not in self.dict_mutation_.keys():
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_,
                                             end_col_offset_)] = []  # a list of mutated ast , not mutated string

                if isinstance(node, ast.Compare):
                    origin_ops_list = node.ops  # it is a list of op
                    # for Conditionals Boundary Mutator
                    for i in range(len(node.ops)):
                        if isinstance(node.ops[i], ast.Lt):
                            i_mutated_ops_list = origin_ops_list.copy()
                            i_mutated_ops_list[i] = ast.LtE()
                            mutated_node_i = ast.Compare(left=copy.deepcopy(node.left), ops=i_mutated_ops_list,
                                                         comparators=copy.deepcopy(node.comparators))
                            ast.copy_location(mutated_node_i, node)
                            mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'CONDITIONALS-BOUNDARY'}
                            self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                                mutated_i_dict)
                        elif isinstance(node.ops[i], ast.LtE):
                            i_mutated_ops_list = origin_ops_list.copy()
                            i_mutated_ops_list[i] = ast.Lt()
                            mutated_node_i = ast.Compare(left=copy.deepcopy(node.left), ops=i_mutated_ops_list,
                                                         comparators=copy.deepcopy(node.comparators))
                            ast.copy_location(mutated_node_i, node)
                            mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'CONDITIONALS-BOUNDARY'}
                            self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                                mutated_i_dict)
                        elif isinstance(node.ops[i], ast.Gt):
                            i_mutated_ops_list = origin_ops_list.copy()
                            i_mutated_ops_list[i] = ast.GtE()
                            mutated_node_i = ast.Compare(left=copy.deepcopy(node.left), ops=i_mutated_ops_list,
                                                         comparators=copy.deepcopy(node.comparators))
                            ast.copy_location(mutated_node_i, node)
                            mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'CONDITIONALS-BOUNDARY'}
                            self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                                mutated_i_dict)
                        elif isinstance(node.ops[i], ast.GtE):
                            i_mutated_ops_list = origin_ops_list.copy()
                            i_mutated_ops_list[i] = ast.Gt()
                            mutated_node_i = ast.Compare(left=copy.deepcopy(node.left), ops=i_mutated_ops_list,
                                                         comparators=copy.deepcopy(node.comparators))
                            ast.copy_location(mutated_node_i, node)
                            mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'CONDITIONALS-BOUNDARY'}
                            self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                                mutated_i_dict)

                    for i in range(len(node.ops)):
                        if isinstance(node.ops[i], ast.Eq):
                            i_mutated_ops_list = origin_ops_list.copy()
                            i_mutated_ops_list[i] = ast.NotEq()
                            mutated_node_i = ast.Compare(left=copy.deepcopy(node.left), ops=i_mutated_ops_list,
                                                         comparators=copy.deepcopy(node.comparators))
                            ast.copy_location(mutated_node_i, node)
                            mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'NEGATE-CONDITIONALS'}
                            self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                                mutated_i_dict)
                        elif isinstance(node.ops[i], ast.NotEq):
                            i_mutated_ops_list = origin_ops_list.copy()
                            i_mutated_ops_list[i] = ast.Eq()
                            mutated_node_i = ast.Compare(left=copy.deepcopy(node.left), ops=i_mutated_ops_list,
                                                         comparators=copy.deepcopy(node.comparators))
                            ast.copy_location(mutated_node_i, node)
                            mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'NEGATE-CONDITIONALS'}
                            self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                                mutated_i_dict)
                        elif isinstance(node.ops[i], ast.LtE):
                            i_mutated_ops_list = origin_ops_list.copy()
                            i_mutated_ops_list[i] = ast.Gt()
                            mutated_node_i = ast.Compare(left=node.left, ops=i_mutated_ops_list,
                                                         comparators=node.comparators)
                            ast.copy_location(mutated_node_i, node)
                            mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'NEGATE-CONDITIONALS'}
                            self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                                mutated_i_dict)
                        elif isinstance(node.ops[i], ast.GtE):
                            i_mutated_ops_list = origin_ops_list.copy()
                            i_mutated_ops_list[i] = ast.Lt()
                            mutated_node_i = ast.Compare(left=copy.deepcopy(node.left), ops=i_mutated_ops_list,
                                                         comparators=copy.deepcopy(node.comparators))
                            ast.copy_location(mutated_node_i, node)
                            mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'NEGATE-CONDITIONALS'}
                            self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                                mutated_i_dict)
                        elif isinstance(node.ops[i], ast.Lt):
                            i_mutated_ops_list = origin_ops_list.copy()
                            i_mutated_ops_list[i] = ast.GtE()
                            mutated_node_i = ast.Compare(left=copy.deepcopy(node.left), ops=i_mutated_ops_list,
                                                         comparators=copy.deepcopy(node.comparators))
                            ast.copy_location(mutated_node_i, node)
                            mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'NEGATE-CONDITIONALS'}
                            self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                                mutated_i_dict)
                        elif isinstance(node.ops[i], ast.Gt):
                            i_mutated_ops_list = origin_ops_list.copy()
                            i_mutated_ops_list[i] = ast.LtE()
                            mutated_node_i = ast.Compare(left=copy.deepcopy(node.left), ops=i_mutated_ops_list,
                                                         comparators=copy.deepcopy(node.comparators))
                            ast.copy_location(mutated_node_i, node)
                            mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'NEGATE-CONDITIONALS'}
                            self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                                mutated_i_dict)

                elif isinstance(node, ast.AugAssign):  # INCREMENTS
                    if isinstance(node.op, ast.Add):
                        mutated_node_i = ast.AugAssign(target=copy.deepcopy(node.target), op=ast.Sub(),
                                                       value=copy.deepcopy(node.value))
                        ast.copy_location(mutated_node_i, node)
                        mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'INCREMENTS'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(mutated_i_dict)
                    elif isinstance(node.op, ast.Sub):
                        mutated_node_i = ast.AugAssign(target=copy.deepcopy(node.target), op=ast.Add(),
                                                       value=copy.deepcopy(node.value))
                        ast.copy_location(mutated_node_i, node)
                        mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'INCREMENTS'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(mutated_i_dict)
                elif isinstance(node, ast.UnaryOp):  # INVERT-NEGS
                    if isinstance(node.op, ast.USub):
                        mutated_node_i = node.operand  # Notice!!!!!
                        mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'INVERT-NEGS'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(mutated_i_dict)

                elif isinstance(node, ast.BinOp):
                    if isinstance(node.op, ast.Add):
                        mutated_node_i = ast.BinOp(left=copy.deepcopy(node.left), op=ast.Sub(),
                                                   right=copy.deepcopy(node.right))
                        ast.copy_location(mutated_node_i, node)
                        mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'MATH'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(mutated_i_dict)
                    elif isinstance(node.op, ast.Sub):
                        mutated_node_i = ast.BinOp(left=copy.deepcopy(node.left), op=ast.Add(),
                                                   right=copy.deepcopy(node.right))
                        ast.copy_location(mutated_node_i, node)
                        mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'MATH'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(mutated_i_dict)
                    elif isinstance(node.op, ast.Mult):
                        mutated_node_i = ast.BinOp(left=copy.deepcopy(node.left), op=ast.Div(),
                                                   right=copy.deepcopy(node.right))
                        ast.copy_location(mutated_node_i, node)
                        mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'MATH'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(mutated_i_dict)
                    elif isinstance(node.op, ast.Div):
                        mutated_node_i = ast.BinOp(left=copy.deepcopy(node.left), op=ast.Mult(),
                                                   right=copy.deepcopy(node.right))
                        ast.copy_location(mutated_node_i, node)
                        mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'MATH'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(mutated_i_dict)
                    elif isinstance(node.op, ast.Mod):
                        mutated_node_i = ast.BinOp(left=copy.deepcopy(node.left), op=ast.Mult(),
                                                   right=copy.deepcopy(node.right))
                        ast.copy_location(mutated_node_i, node)
                        mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'MATH'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(mutated_i_dict)
                    elif isinstance(node.op, ast.BitAnd):
                        # MATH
                        mutated_node_i = ast.BinOp(left=copy.deepcopy(node.left), op=ast.BitOr(),
                                                   right=copy.deepcopy(node.right))
                        ast.copy_location(mutated_node_i, node)
                        mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'MATH'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(mutated_i_dict)

                        # OBBN1
                        mutated_OBBN1_i = ast.BinOp(left=copy.deepcopy(node.left), op=ast.BitOr(),
                                                    right=copy.deepcopy(node.right))
                        ast.copy_location(mutated_OBBN1_i, node)
                        mutated_i_OBBN1_dict = {'mutated_ast': mutated_OBBN1_i, 'type': 'OBBN1'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                            mutated_i_OBBN1_dict)

                        # OBBN2
                        mutated_OBBN2_i = copy.deepcopy(node.left)  # Notice!!!!
                        mutated_i_OBBN2_dict = {'mutated_ast': mutated_OBBN2_i, 'type': 'OBBN2'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                            mutated_i_OBBN2_dict)

                        # OBBN3
                        mutated_OBBN3_i = copy.deepcopy(node.right)  # Notice!!!!
                        mutated_i_OBBN3_dict = {'mutated_ast': mutated_OBBN3_i, 'type': 'OBBN3'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                            mutated_i_OBBN3_dict)

                    elif isinstance(node.op, ast.BitOr):
                        # MATH
                        mutated_node_i = ast.BinOp(left=copy.deepcopy(node.left), op=ast.BitAnd(),
                                                   right=copy.deepcopy(node.right))
                        ast.copy_location(mutated_node_i, node)
                        mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'MATH'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(mutated_i_dict)

                        # OBBN1
                        mutated_OBBN1_i = ast.BinOp(left=copy.deepcopy(node.left), op=ast.BitAnd(),
                                                    right=copy.deepcopy(node.right))
                        ast.copy_location(mutated_OBBN1_i, node)
                        mutated_i_OBBN1_dict = {'mutated_ast': mutated_OBBN1_i, 'type': 'OBBN1'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                            mutated_i_OBBN1_dict)

                        # OBBN2
                        mutated_OBBN2_i = copy.deepcopy(node.left)  # notice!!!!
                        mutated_i_OBBN2_dict = {'mutated_ast': mutated_OBBN2_i, 'type': 'OBBN2'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                            mutated_i_OBBN2_dict)

                        # OBBN3
                        mutated_OBBN3_i = copy.deepcopy(node.right)  # notice!!!!
                        mutated_i_OBBN3_dict = {'mutated_ast': mutated_OBBN3_i, 'type': 'OBBN3'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                            mutated_i_OBBN3_dict)

                    elif isinstance(node.op, ast.BitXor):
                        mutated_node_i = ast.BinOp(left=copy.deepcopy(node.left), op=ast.BitAnd(),
                                                   right=copy.deepcopy(node.right))
                        ast.copy_location(mutated_node_i, node)
                        mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'MATH'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(mutated_i_dict)
                    elif isinstance(node.op, ast.LShift):
                        mutated_node_i = ast.BinOp(left=copy.deepcopy(node.left), op=ast.RShift(),
                                                   right=copy.deepcopy(node.right))
                        ast.copy_location(mutated_node_i, node)
                        mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'MATH'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(mutated_i_dict)
                    elif isinstance(node.op, ast.RShift):
                        mutated_node_i = ast.BinOp(left=copy.deepcopy(node.left), op=ast.LShift(),
                                                   right=copy.deepcopy(node.right))
                        ast.copy_location(mutated_node_i, node)
                        mutated_i_dict = {'mutated_ast': mutated_node_i, 'type': 'MATH'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(mutated_i_dict)

                elif isinstance(node, ast.Return):
                    added_False = 1
                    added_True = 1
                    added_None = 1
                    try:
                        if isinstance(node.value, ast.Constant):
                            if (node.value.value == True) and (node.value.kind == None):
                                added_True -= 1
                            if (node.value.value == False) and (node.value.kind == None):
                                added_False -= 1
                            if (node.value.value == None) and (node.value.kind == None):
                                added_None -= 1
                    except:
                        pass
                    if added_False == 1:
                        mutated_node_i_False = ast.Return(value=ast.Constant(value=False, kind=None))
                        ast.copy_location(mutated_node_i_False, node)
                        mutated_i_dict_False = {'mutated_ast': mutated_node_i_False, 'type': 'FALSE-RETURNS'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                            mutated_i_dict_False)

                    if added_True == 1:
                        mutated_node_i_True = ast.Return(value=ast.Constant(value=True, kind=None))
                        ast.copy_location(mutated_node_i_True, node)
                        mutated_i_dict_True = {'mutated_ast': mutated_node_i_True, 'type': 'TRUE-RETURNS'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                            mutated_i_dict_True)

                    if added_None == 1:
                        mutated_node_i_None = ast.Return(value=ast.Constant(value=None, kind=None))
                        ast.copy_location(mutated_node_i_None, node)
                        mutated_i_dict_None = {'mutated_ast': mutated_node_i_None, 'type': 'NULL-RETURNS'}
                        self.dict_mutation_[(lineno_, col_offset_, end_lineno_, end_col_offset_)].append(
                            mutated_i_dict_None)

                elif isinstance(node, ast.Assign):  # CRCR
                    if isinstance(node.value, ast.Constant):
                        value_lineno_, value_col_offset_, value_end_lineno_, value_end_col_offset_ = node.value.lineno, node.value.col_offset, node.value.end_lineno, node.value.end_col_offset
                        if (value_lineno_, value_col_offset_, value_end_lineno_,
                            value_end_col_offset_) not in self.dict_mutation_.keys():
                            self.dict_mutation_[
                                (value_lineno_, value_col_offset_, value_end_lineno_, value_end_col_offset_)] = []

                        added_CRCR1 = 1
                        added_CRCR2 = 1
                        if isinstance(node.value.value, (int, float)):
                            if node.value.value == 1:
                                added_CRCR1 -= 1
                            if node.value.value == 0:
                                added_CRCR2 -= 1

                        if added_CRCR1 == 1:
                            mutated_CRCR1 = ast.Constant(value=1, kind=None)
                            ast.copy_location(mutated_CRCR1, node.value)
                            mutated_CRCR1_dict = {'mutated_ast': mutated_CRCR1, 'type': 'CRCR1'}
                            self.dict_mutation_[
                                (value_lineno_, value_col_offset_, value_end_lineno_, value_end_col_offset_)].append(
                                mutated_CRCR1_dict)
                        if added_CRCR2 == 1:
                            mutated_CRCR2 = ast.Constant(value=0, kind=None)
                            ast.copy_location(mutated_CRCR2, node.value)
                            mutated_CRCR2_dict = {'mutated_ast': mutated_CRCR2, 'type': 'CRCR2'}
                            self.dict_mutation_[
                                (value_lineno_, value_col_offset_, value_end_lineno_, value_end_col_offset_)].append(
                                mutated_CRCR2_dict)

                        mutated_CRCR3 = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1, kind=None))
                        ast.copy_location(mutated_CRCR3, node.value)
                        mutated_CRCR3_dict = {'mutated_ast': mutated_CRCR3, 'type': 'CRCR3'}
                        self.dict_mutation_[
                            (value_lineno_, value_col_offset_, value_end_lineno_, value_end_col_offset_)].append(
                            mutated_CRCR3_dict)

                        # CRCR4, CRCR5, CRCR6
                        if isinstance(node.value.value, (int, float)):
                            mutated_CRCR4 = ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(node.value))
                            ast.copy_location(mutated_CRCR4, node.value)
                            mutated_CRCR4_dict = {'mutated_ast': mutated_CRCR4, 'type': 'CRCR4'}
                            self.dict_mutation_[
                                (value_lineno_, value_col_offset_, value_end_lineno_, value_end_col_offset_)].append(
                                mutated_CRCR4_dict)

                            mutated_CRCR5 = ast.BinOp(left=copy.deepcopy(node.value), op=ast.Add(),
                                                      right=ast.Constant(value=1, kind=None))
                            ast.copy_location(mutated_CRCR5, node.value)
                            mutated_CRCR5_dict = {'mutated_ast': mutated_CRCR5, 'type': 'CRCR5'}
                            self.dict_mutation_[
                                (value_lineno_, value_col_offset_, value_end_lineno_, value_end_col_offset_)].append(
                                mutated_CRCR5_dict)

                            mutated_CRCR6 = ast.BinOp(left=copy.deepcopy(node.value), op=ast.Sub(),
                                                      right=ast.Constant(value=1, kind=None))
                            ast.copy_location(mutated_CRCR6, node.value)
                            mutated_CRCR6_dict = {'mutated_ast': mutated_CRCR6, 'type': 'CRCR6'}
                            self.dict_mutation_[
                                (value_lineno_, value_col_offset_, value_end_lineno_, value_end_col_offset_)].append(
                                mutated_CRCR6_dict)

                ast.NodeVisitor.generic_visit(self, node)

    class Modify_tree(ast.NodeTransformer):
        def __init__(self, new_node, lineno, col_offset, end_lineno, end_col_offset):  # new_node is an AST
            self.new_node = copy.deepcopy(new_node)
            self.lineno = lineno
            self.col_offset = col_offset
            self.end_lineno = end_lineno
            self.end_col_offset = end_col_offset

        def generic_visit(self, node):
            ast.NodeTransformer.generic_visit(self, node)
            try:
                lineno_, col_offset_, end_lineno_, end_col_offset_ = node.lineno, node.col_offset, node.end_lineno, node.end_col_offset
            except:
                return node
            else:
                if lineno_ == self.lineno and col_offset_ == self.col_offset and end_lineno_ == self.end_lineno and end_col_offset_ == self.end_col_offset:
                    ast.copy_location(self.new_node, node)
                    return self.new_node
                else:
                    return node

    class Generate_mutants(ast.NodeVisitor):
        def __init__(self, origin_root, dict_mutations_, filepath, list_text_lines, collect_mutants):
            self.origin_root = copy.deepcopy(origin_root)
            self.dict_mutations = dict_mutations_
            self.filepath = filepath
            self.list_text_lines = list_text_lines
            self.collect_mutants = collect_mutants

        def generic_visit(self, node):
            try:
                lineno_, col_offset_, end_lineno_, end_col_offset_ = node.lineno, node.col_offset, node.end_lineno, node.end_col_offset
            except:
                ast.NodeVisitor.generic_visit(self, node)
            else:
                if (lineno_, col_offset_, end_lineno_, end_col_offset_) in self.dict_mutations.keys():
                    list_mutant_dicts_ = self.dict_mutations[(lineno_, col_offset_, end_lineno_, end_col_offset_)]
                    # list_mutant_dicts_ = [{'mutated_ast': <ast.Compare object at 0x7efec08695e0>, 'type': 'CONDITIONALS-BOUNDARY'}, {'mutated_ast': <ast.Compare object at 0x7efec0869640>, 'type': 'NEGATE-CONDITIONALS'}]
                    for mutant_dict in list_mutant_dicts_:
                        mutant_type = mutant_dict['type']
                        M_tree = Modify_tree(mutant_dict['mutated_ast'], lineno_, col_offset_, end_lineno_,end_col_offset_)
                        copy_origin_root = copy.deepcopy(self.origin_root)
                        mutated_root = M_tree.visit(copy_origin_root)
                        mutated_root_str = ast.unparse(mutated_root)
                        mutated_root_split_str_list = mutated_root_str.split("\n")

                        mutated_line_ = mutated_root_split_str_list[lineno_ - 1]

                        self.collect_mutants.append(
                            {'lineno': lineno_, 'col_offset': col_offset_, 'mutated_root_str': mutated_root_str,
                             'mutated_line': mutated_line_,'type': mutant_type, 'filepath': self.filepath})
                ast.NodeVisitor.generic_visit(self, node)

    # collect_mutants:  [{'lineno': 2, 'col_offset': 7, 'mutated_root': <ast.Module object at 0x7f8ab1be7b20>, 'mutated_list_text_lines': [...] ,'mutated_line': '    if a < b and c > d:', 'type': 'CONDITIONALS-BOUNDARY', 'filepath': 'target/bar/law.py'},,...]
    def Generate_diff(collect_mutant_roots, mutant_dir, mode_mutate):  # only for a file
        import tempfile
        processed_mutant_roots = {}
        for mutant_dict in collect_mutant_roots:
            if mutant_dict['type'] not in processed_mutant_roots.keys():
                processed_mutant_roots[mutant_dict['type']] = {}
        for mutant_dict in collect_mutant_roots:
            mutant_type = mutant_dict['type']
            mutant_lineno = mutant_dict['lineno']
            if mutant_dict['lineno'] not in processed_mutant_roots[mutant_type].keys():
                processed_mutant_roots[mutant_type][mutant_lineno] = []
        for mutant_dict in collect_mutant_roots:
            mutant_type = mutant_dict['type']
            mutant_lineno = mutant_dict['lineno']
            for_append_ = {'mutated_root_str': mutant_dict['mutated_root_str'],
                           'mutated_line': mutant_dict['mutated_line'], 'filepath': mutant_dict['filepath']}
            processed_mutant_roots[mutant_type][mutant_lineno].append(for_append_)
        for mutant_type in processed_mutant_roots.keys():
            for mutant_lineno in processed_mutant_roots[mutant_type].keys():
                processed_mutant_roots[mutant_type][mutant_lineno] = sorted(
                    processed_mutant_roots[mutant_type][mutant_lineno], key=lambda x: x['mutated_line'])
        for mutant_type in processed_mutant_roots.keys():
            for mutant_lineno in processed_mutant_roots[mutant_type].keys():
                for i in range(len(processed_mutant_roots[mutant_type][mutant_lineno])):
                    processed_mutant_roots[mutant_type][mutant_lineno][i]['index'] = '0' + str(i) if i < 10 else str(i)
        count_generated_mutants = 0
        list_collect_mutants_ = []
        # generate .diff
        for mutant_type in processed_mutant_roots.keys():
            for mutant_lineno in processed_mutant_roots[mutant_type].keys():
                for i in range(len(processed_mutant_roots[mutant_type][mutant_lineno])):
                    dict_i = processed_mutant_roots[mutant_type][mutant_lineno][i]
                    src_filepath = dict_i['filepath']
                    src_filename = src_filepath.split('/')[-1]  # src_filename = 'law.py'
                    mutant_lineno_str = '0' + str(mutant_lineno) if mutant_lineno < 10 else str(mutant_lineno)
                    diff_filename = src_filename.split('.')[0] + '_' + mutant_type + '_' + mutant_lineno_str + '_' + dict_i['index'] + '.diff'

                    temp = tempfile.NamedTemporaryFile(suffix='.py')
                    mutated_root_string = dict_i['mutated_root_str']
                    mutated_root_byte = mutated_root_string.encode('utf-8')
                    temp.write(mutated_root_byte)
                    temp.seek(0)
                    temp_filepath = temp.name

                    if mode_mutate == True:
                        diff_filepath = os.path.join(mutant_dir, diff_filename)
                        os.system("diff " + src_filepath + " " + temp_filepath + " > " + diff_filepath)
                    temp.close()

                    count_generated_mutants += 1
                    list_collect_mutants_.append([diff_filename.split(".")[0], mutated_root_string])
        return count_generated_mutants, list_collect_mutants_

    def Modify_testfile(filepath): #does not modify file # valid
        test_file_lines_ = open(filepath, "r").readlines()
        test_file_ast_ = ast.parse("".join(test_file_lines_), filepath)
        last_node_ast = test_file_ast_.body[-1]

        list_test_function = []
        for i in range(len(test_file_ast_.body)):
            if isinstance(test_file_ast_.body[i], ast.FunctionDef):
                function_name_ = test_file_ast_.body[i].name
                if function_name_.startswith("test"):
                    list_test_function.append(function_name_)

        for i in range(len(list_test_function)):
            body_branch = ast.parse("func()").body[0]
            body_branch.value.func.id = list_test_function[i]
            ast.copy_location(body_branch, last_node_ast)

            except_kill = ast.parse("KM.dict_['func'] = 1").body[0]
            except_kill.targets[0].slice.value = list_test_function[i]
            ast.copy_location(except_kill, last_node_ast)
            handler_except = ast.ExceptHandler(type=None, name=None, body=[except_kill])
            ast.copy_location(handler_except, last_node_ast)

            else_kill = ast.parse("KM.dict_['func'] = 0").body[0]
            else_kill.targets[0].slice.value = list_test_function[i]
            ast.copy_location(else_kill, last_node_ast)

            try_node = ast.Try(body=[body_branch], handlers = [handler_except], orelse=[else_kill], finalbody=[])
            ast.copy_location(try_node, last_node_ast)

            test_file_ast_.body.append(try_node)
        str_modified_testfile = astor.to_source(test_file_ast_)
        return str_modified_testfile # a string

    def Modify_code(filepath, code_str): # modify file
        with open(filepath, "w") as file:
            file.write(code_str)
            file.close()

    # python3 pmut.py --action mutate --source target/bar --mutants nhap/Mutants/bar
    if args.action == "mutate":
        os.makedirs(args.mutants, exist_ok=True)

        list_source_filepaths_tmp = []
        for filename in os.listdir(args.source):
            filepath = os.path.join(args.source, filename)
            list_source_filepaths_tmp.append(filepath)
        list_source_filepaths = [f_path for f_path in list_source_filepaths_tmp if os.path.isfile(f_path)]

        num_mutated_file = 0
        num_generated_mutants = 0
        for src_filepath in list_source_filepaths:
            src_lines = open(src_filepath, "r").readlines()
            root = ast.parse("".join(src_lines), src_filepath)

            dict_mutation = {}
            V_mutate = Visit_mutate_tree(dict_mutation_=dict_mutation)
            V_mutate.visit(root)
            # dict_mutation now has been updated
            # (self, origin_root, dict_mutations_, filepath, collect_mutants):
            collect_mutants = []
            Generate_Tree = Generate_mutants(origin_root=root, dict_mutations_=dict_mutation, filepath=src_filepath,
                                             list_text_lines=src_lines, collect_mutants=collect_mutants)
            Generate_Tree.visit(copy.deepcopy(root))
            # collect_mutants has been updated

            # collect_mutants:  [{'lineno': 2, 'col_offset': 7, 'mutated_root': <ast.Module object at 0x7fdad0fccd90>, 'type': 'CONDITIONALS-BOUNDARY', 'filepath': 'target/bar/law.py'},,...]
            # Generate_diff(collect_mutant_roots, mutant_dir)
            count_file_generated_mutants, list_collect_mutants_ = Generate_diff(collect_mutants, args.mutants, mode_mutate=True)
            # list_collect_mutants_ is a list of pairs [diff_filename.split(".")[0], mutated_root_string]
            if count_file_generated_mutants > 0:
                num_mutated_file += 1
            num_generated_mutants += count_file_generated_mutants

        print("Total number of mutated files: ", num_mutated_file)
        print("Total number of mutants generated: ", num_generated_mutants)
        # store_mutant_flag_dicts =  {'target/bar/law.py': [['law_CONDITIONALS-BOUNDARY_02_00', 'def bar1(a, b, c, d):\n    if a < b and c > d:\n ...]..[.]..}
        # len(store_mutant_flag_dicts['target/bar/law.py'] = 60
        # store_mutant_flag_dicts['target/bar/law.py'] is a list of pairs [diff_filename.split(".")[0], mutated_root_string]

    elif args.action == "execute":
        os.makedirs(args.kill, exist_ok=True)
        ######################### GENERATE MUTANTS
        list_source_filepaths_tmp = []
        for filename in os.listdir(args.source):
            filepath = os.path.join(args.source, filename)
            list_source_filepaths_tmp.append(filepath)
        list_source_filepaths = [f_path for f_path in list_source_filepaths_tmp if os.path.isfile(f_path)]
        # list_source_filepaths = ['./target/bar/law.py', './target/bar/simple.py', './target/bar/lazy.py']

        original_code_source_files = {}
        for src_filepath in list_source_filepaths:
            src_lines_ = open(src_filepath, "r").readlines()
            original_code_source_files[src_filepath] = "".join(src_lines_)

        all_mutants = {}
        if args.mutants == None:
            for src_filepath in list_source_filepaths:
                src_lines = open(src_filepath, "r").readlines()
                root = ast.parse("".join(src_lines), src_filepath)

                dict_mutation = {}
                V_mutate = Visit_mutate_tree(dict_mutation_=dict_mutation)
                V_mutate.visit(root)
                # dict_mutation now has been updated
                collect_mutants = []
                Generate_Tree = Generate_mutants(origin_root=root, dict_mutations_=dict_mutation, filepath=src_filepath, list_text_lines=src_lines, collect_mutants=collect_mutants)
                Generate_Tree.visit(copy.deepcopy(root))
                # collect_mutants has been updated
                # collect_mutants:  [{'lineno': 2, 'col_offset': 7, 'mutated_root': <ast.Module object at 0x7fdad0fccd90>, 'type': 'CONDITIONALS-BOUNDARY', 'filepath': 'target/bar/law.py'},,...]
                _ , list_collect_mutants_ = Generate_diff(collect_mutants, args.mutants, mode_mutate=False)
                # list_collect_mutants_ is a list of pairs [diff_filename.split(".")[0], mutated_root_string]
                all_mutants[src_filepath] = list_collect_mutants_.copy()
        else:
            list_diff_files = os.listdir(args.mutants)
            for diff_filename in list_diff_files:
                diff_filepath = os.path.join(args.mutants, diff_filename)
                target_filename = diff_filename.split("_")[0] + ".py"
                target_filepath = os.path.join(args.source, target_filename)
                original_target_file_code_str = original_code_source_files[target_filepath]
                ## Error prone
                # patching
                os.system("patch --silent {} < {}".format(target_filepath, diff_filepath))
                with open(target_filepath, "r") as file_:
                    mutant_code_str = file_.readlines()
                mutant_code_str = "".join(mutant_code_str)

                if target_filepath not in all_mutants.keys():
                    all_mutants[target_filepath] = []
                all_mutants[target_filepath].append([diff_filename.split(".")[0], mutant_code_str])
                Modify_code(target_filepath, original_target_file_code_str)

        # all_mutants =  {'./target/bar/law.py': [['law_CONDITIONALS-BOUNDARY_02_00', 'def bar1(a, b, c, d):\n    if a < b and c > d:\n...']...],...}
        # all_mutants[src_filepath] is a list of pairs [diff_filename.split(".")[0], mutated_root_string]
        mutant_json = {}
        for filepath in all_mutants.keys():
            for pair_ in all_mutants[filepath]:
                current_num_keys_mutant_json = len(list(mutant_json.keys()))
                mutant_json[pair_[0]] = current_num_keys_mutant_json

        ######################### collect TEST
        list_testfile_paths = []
        for root_dir, dirs, files in os.walk(args.source):
            for file in files:
                if (file.startswith("test_") and file.endswith(".py")) or file.endswith("_test.py"):
                    list_testfile_paths.append(os.path.join(root_dir, file))

        collect_test_functions = {}
        for testfile_path in list_testfile_paths:
            collect_test_functions[testfile_path] = []
        for testfile_path in list_testfile_paths:
            testfile_lines = open(testfile_path, "r").readlines()
            testfile_ast_ = ast.parse("".join(testfile_lines), testfile_path)
            for i in range(len(testfile_ast_.body)):
                if isinstance(testfile_ast_.body[i], ast.FunctionDef):
                    function_name_ = testfile_ast_.body[i].name
                    if function_name_.startswith("test"):
                        collect_test_functions[testfile_path].append(function_name_)
        # collect_test_functions:  {'./target/bar/testing/test_bar_lazy.py': ['test_sort1', 'test_sort2', 'test_sort3'], ...}
        test_json = {}
        for testfile_path in collect_test_functions.keys():
            testfile_name_without_py = testfile_path.split("/")[-1].split(".")[0]
            for test_function_name in collect_test_functions[testfile_path]:
                num_current_keys_test_json = len(list(test_json.keys()))
                index_name = test_function_name + "@" + testfile_name_without_py
                test_json[index_name] = num_current_keys_test_json

        num_test_found = 0
        for testfile_path_ in collect_test_functions.keys():
            num_test_found += len(collect_test_functions[testfile_path_])
        print("Total test functions found: ", num_test_found)
        num_mutant_generated = 0
        for file_path_ in all_mutants.keys():
            num_mutant_generated += len(all_mutants[file_path_])

        ######################### kill mutants
        # Kill Matrix:[number of test cases] by [number of mutants]
        import numpy as np
        kill_matrix = np.zeros(num_test_found*num_mutant_generated, dtype='int32')
        kill_matrix = kill_matrix.reshape((num_test_found, num_mutant_generated))
        root_path = os.getcwd() #  root_path = '/home/dat/KAIST learning/Fourth semester/Automatic software Testing/Homeworks and assignment/Homework3/cs453-python-mutation-testing-tiendatnguyen-vision'

        original_code_testfiles = {}
        for testfile_path in list_testfile_paths:
            testfile_lines = open(testfile_path, "r").readlines()
            original_code_testfiles[testfile_path] = "".join(testfile_lines)

        # Now we collect the changed strings of test files
        dict_modified_testfiles = {}
        for testfile_path in list_testfile_paths:
            str_modified_testfile = Modify_testfile(testfile_path)
            dict_modified_testfiles[testfile_path] = str_modified_testfile
        # all_mutants =  {'./target/bar/law.py': [['law_CONDITIONALS-BOUNDARY_02_00', 'def bar1(a, b, c, d):\n    if a < b and c > d:\n...']...],...}
        # all_mutants[src_filepath] is a list of pairs [diff_filename.split(".")[0], mutated_root_string]

        before_exec_module = list(sys.modules.keys()).copy()
        for src_filepath in list_source_filepaths:
            list_src_mutants = all_mutants[src_filepath]
            for src_mutant in list_src_mutants: #Modify_code(filepath, code_str)
                Modify_code(src_filepath, src_mutant[1])
                mutant_name = src_mutant[0]
                matrix_column_index = mutant_json[mutant_name]
                # file has been mutated
                class KILL_MUTANTS():
                    dict_ = {}
                KM = KILL_MUTANTS()
                # execute test files to kill mutants
                # collect_test_functions:  {'./target/bar/testing/test_bar_lazy.py': ['test_sort1', 'test_sort2', 'test_sort3'], ...}
                for i in range(len(list_testfile_paths)):
                    os.chdir(root_path)
                    test_file_path_ = list_testfile_paths[i]
                    list_functions_i = collect_test_functions[test_file_path_] # ['test_sort1', 'test_sort2', 'test_sort3']
                    test_file_abspath_ = os.path.abspath(test_file_path_)
                    os.chdir(os.path.dirname(test_file_abspath_))
                    # bug here
                    #before_exec_module = list(sys.modules.keys()).copy()
                    exec(dict_modified_testfiles[test_file_path_], {"KM": KM})
                    after_exec_module = list(sys.modules.keys()).copy()
                    for key_ in after_exec_module:
                        if key_ not in before_exec_module:
                            sys.modules.pop(key_, None)

                    for j in range(len(list_functions_i)):
                        test_index_name = list_functions_i[j] + "@" + test_file_path_.split("/")[-1].split(".")[0]
                        matrix_row_index = test_json[test_index_name]
                        kill_matrix[matrix_row_index][matrix_column_index] = KM.dict_[list_functions_i[j]]
                    os.chdir(root_path)
                # Recover the src_file
                original_code_src_ = original_code_source_files[src_filepath]
                Modify_code(src_filepath, original_code_src_)
        # Now we recover the original content of source files and test files
        for src_filepath in list_source_filepaths:
            original_code = original_code_source_files[src_filepath]
            with open(src_filepath, "w") as file_:
                file_.write(original_code)
                file_.close()
        for testfile_path in list_testfile_paths:
            original_code_ = original_code_testfiles[testfile_path]
            with open(testfile_path, "w") as file_:
                file_.write(original_code_)
                file_.close()

        # print result
        num_killed_mutants = 0
        tmp_matrix = np.array([0 for _ in range(num_mutant_generated)])
        for i in range(num_test_found):
            for j in range(num_mutant_generated):
                tmp_matrix[j] += kill_matrix[i][j]
        for j in range(num_mutant_generated):
            if tmp_matrix[j] > 0:
                num_killed_mutants += 1
        print("Total killed mutants: ", num_killed_mutants)
        num_mutants_ = kill_matrix.shape[1]
        mutation_score = 100*num_killed_mutants/num_mutants_
        print("Mutation Score: {0:.2f}%".format(mutation_score) + " ({} / {})".format(num_killed_mutants, num_mutants_))
        dict_tmp_matrix = {}
        for i in tmp_matrix.tolist():
            if i not in dict_tmp_matrix.keys():
                dict_tmp_matrix[i] =1
            else:
                dict_tmp_matrix[i] += 1

        # write to files
        import json
        with open(os.path.join(args.kill, "test_index.json"), "w") as json_file:
            json.dump(test_json, json_file)
        with open(os.path.join(args.kill, "mutation_index.json"), "w") as json_file:
            json.dump(mutant_json, json_file)
        np.savetxt(fname=os.path.join(args.kill, "kill_matrix.np"), X=kill_matrix, fmt='%.1u')
