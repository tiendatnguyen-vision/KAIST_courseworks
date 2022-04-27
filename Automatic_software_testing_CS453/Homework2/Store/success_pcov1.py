import ast
import argparse
import sys
from copy import deepcopy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measures coverage.')
    parser.add_argument('-v', '--verbose', action="store_true")
    parser.add_argument('-t', '--target', required=True)
    parser.add_argument("remaining", nargs="*")
    args = parser.parse_args()

    target = args.target
    lines = open(target, "r").readlines()
    root = ast.parse("".join(lines), target)


    # instrument the target script
    # my code from here
    def merge_line(line_list, start_row, start_col, end_row, end_col):
        if start_row == end_row:
            return line_list[start_row - 1][start_col:end_col]
        else:
            s = line_list[start_row - 1][start_col:]
            for i in range(start_row, end_row - 1):
                s += line_list[i]
            s += line_list[end_row - 1][:end_col]
            return s


    def help_Ifexp(predicate, dict_):
        value_predicate = predicate
        dict_['true'] += int(value_predicate)
        dict_['false'] += 1 - int(value_predicate)
        return value_predicate


    def help_modify_condition(predicate, dict_):
        value_predicate = predicate
        dict_['true'] += int(value_predicate)
        dict_['false'] += 1 - int(value_predicate)
        return value_predicate


    class Statement_dic():
        dict_enter = {}


    S = Statement_dic()


    class Branch_dic():
        dict_if = {}


    B = Branch_dic()


    class Condition_dic():
        dict_con = {}


    C = Condition_dic()


    class Initial_visit_tree(ast.NodeVisitor):
        def generic_visit(self, node):
            if isinstance(node, ast.stmt):
                lineno_ = node.lineno
                col_offset_ = node.col_offset
                if (lineno_, col_offset_) not in S.dict_enter.keys():
                    S.dict_enter[(lineno_, col_offset_)] = 0
            ast.NodeVisitor.generic_visit(self, node)


    I = Initial_visit_tree()
    I.visit(root)


    class Initial_visit_branch(ast.NodeVisitor):
        def generic_visit(self, node):
            if isinstance(node, ast.If) or isinstance(node, ast.For) or isinstance(node, ast.While) or isinstance(node,
                                                                                                                  ast.AsyncFor) or isinstance(
                    node, ast.IfExp):
                lineno_ = node.lineno
                col_offset_ = node.col_offset
                # find verbose
                if isinstance(node, ast.If):
                    start_r, start_c, end_r, end_c = node.test.lineno, node.test.col_offset, node.test.end_lineno, node.test.end_col_offset
                elif isinstance(node, ast.For) or isinstance(node, ast.AsyncFor):
                    start_r, start_c, end_r, end_c = node.target.lineno, node.target.col_offset, node.iter.end_lineno, node.iter.end_col_offset
                elif isinstance(node, ast.While):
                    start_r, start_c, end_r, end_c = node.test.lineno, node.test.col_offset, node.test.end_lineno, node.test.end_col_offset
                elif isinstance(node, ast.IfExp):
                    start_r, start_c, end_r, end_c = node.test.lineno, node.test.col_offset, node.test.end_lineno, node.test.end_col_offset

                verbose_ = merge_line(lines, start_r, start_c, end_r, end_c)
                if (lineno_, col_offset_) not in B.dict_if.keys():
                    B.dict_if[(lineno_, col_offset_)] = {'true': 0, 'false': 0, 'verbose': verbose_}
            elif isinstance(node, ast.ListComp):
                for generator_ in node.generators:
                    for ifs_ in generator_.ifs:
                        lineno_ = ifs_.lineno
                        col_offset_ = ifs_.col_offset
                        start_r, start_c, end_r, end_c = ifs_.lineno, ifs_.col_offset, ifs_.end_lineno, ifs_.end_col_offset
                        verbose_ = merge_line(lines, start_r, start_c, end_r, end_c)
                        if (lineno_, col_offset_) not in B.dict_if.keys():
                            B.dict_if[(lineno_, col_offset_)] = {'true': 0, 'false': 0, 'verbose': verbose_}
            elif isinstance(node, ast.Try):
                list_handlers = node.handlers  # a list of ast.ExceptHandler
                for handler_ in list_handlers:
                    lineno_handler = handler_.lineno
                    col_offset_handler = handler_.col_offset
                    if handler_.type != None:
                        verbose_ = 'try ==> ' + handler_.type.id
                    else:
                        verbose_ = 'try ===> *'

                    if (lineno_handler, col_offset_handler) not in B.dict_if.keys():
                        B.dict_if[(lineno_handler, col_offset_handler)] = {'executed': 0, 'verbose': verbose_}
            ast.NodeVisitor.generic_visit(self, node)


    I_Branch = Initial_visit_branch()
    I_Branch.visit(root)


    class Initial_visit_condition(ast.NodeVisitor):
        def help_visit_predicate(self, predicate_):
            if isinstance(predicate_, ast.BoolOp):
                for i in range(len(predicate_.values)):
                    self.help_visit_predicate(predicate_.values[i])
            elif isinstance(predicate_, ast.UnaryOp):
                if isinstance(predicate_.op, ast.Not):
                    self.help_visit_predicate(predicate_.operand)
            else:
                if isinstance(predicate_, ast.Compare):
                    if len(predicate_.ops) > 1:
                        start_r, start_c, end_r, end_c = predicate_.lineno, predicate_.col_offset, predicate_.end_lineno, predicate_.end_col_offset
                        list_operands = [predicate_.left] + predicate_.comparators
                        if (start_r, start_c, end_r, end_c) not in C.dict_con.keys():
                            C.dict_con[(start_r, start_c, end_r, end_c)] = {'conjunction': True}
                            for i in range(len(predicate_.ops)):
                                divided_pred = ast.Compare(left=list_operands[i], ops=[predicate_.ops[i]],
                                                           comparators=[list_operands[i + 1]])
                                divided_pred_node = ast.Module(body=[ast.Expr(value=divided_pred)], type_ignores=[])
                                verbose_ = ast.unparse(divided_pred_node)
                                C.dict_con[(start_r, start_c, end_r, end_c)][str(i)] = {'verbose': verbose_, 'true': 0,
                                                                                        'false': 0}
                    else:
                        start_r, start_c, end_r, end_c = predicate_.lineno, predicate_.col_offset, predicate_.end_lineno, predicate_.end_col_offset
                        verbose_ = merge_line(lines, start_r, start_c, end_r, end_c)
                        if (start_r, start_c, end_r, end_c) not in C.dict_con.keys():
                            C.dict_con[(start_r, start_c, end_r, end_c)] = {'conjunction': False, 'verbose': verbose_,
                                                                            'true': 0, 'false': 0}
                else:
                    start_r, start_c, end_r, end_c = predicate_.lineno, predicate_.col_offset, predicate_.end_lineno, predicate_.end_col_offset
                    verbose_ = merge_line(lines, start_r, start_c, end_r, end_c)
                    if (start_r, start_c, end_r, end_c) not in C.dict_con.keys():
                        C.dict_con[(start_r, start_c, end_r, end_c)] = {'conjunction': False, 'verbose': verbose_,
                                                                        'true': 0, 'false': 0}

        def generic_visit(self, node):
            if isinstance(node, ast.If):
                self.help_visit_predicate(node.test)
            elif isinstance(node, ast.While):
                self.help_visit_predicate(node.test)
            elif isinstance(node, ast.IfExp):
                self.help_visit_predicate(node.test)
            elif isinstance(node, ast.ListComp):
                for i in range(len(node.generators)):
                    for j in range(len(node.generators[i].ifs)):
                        self.help_visit_predicate(node.generators[i].ifs[j])
            ast.NodeVisitor.generic_visit(self, node)


    I_Con = Initial_visit_condition()
    I_Con.visit(root)


    def help_visit_Compare(list_operand, list_operator, dict_):
        # dict_ = {'conjunction': True, '0': {'verbose': '', 'true':0, 'false':0}, '1': ...}
        # list_operand = [lambda : 1, lambda :g(), lambda :3]
        # list_operator = [lambda x, y: x!=y]
        first_operand = list_operand[0]()
        list_executed_operand = [first_operand]
        list_value = []
        for i in range(len(list_operand) - 1):
            next_operand = list_operand[i + 1]()
            old_operand = list_executed_operand[-1]
            result = list_operator[i](old_operand, next_operand)
            list_value.append(result)
            list_executed_operand.append(next_operand)
            if result == False:
                for j in range(len(list_value)):
                    dict_[str(j)]['true'] += int(list_value[j])
                    dict_[str(j)]['false'] += 1 - int(list_value[j])
                return False
        return True


    # Now S.dict_enter has been initialized, it's like this:
    # S.dict_enter:  {(1, 0): 0, (3, 0): 0, (4, 0): 0, (6, 0): 0, (7, 4): 0, (8, 0): 0, (9, 4): 0, (11, 4): 0}
    class Rewrite_predicate(ast.NodeTransformer):
        def rewrite_atomic_predicate(self, predicate_):
            start_r, start_c, end_r, end_c = predicate_.lineno, predicate_.col_offset, predicate_.end_lineno, predicate_.end_col_offset
            if isinstance(predicate_, ast.Compare):
                if len(predicate_.ops) > 1:
                    ori_operands = [predicate_.left] + predicate_.comparators  # a list of operands
                    ori_ops = predicate_.ops  # a list of operators
                    list_operand_elts = []
                    list_ops_elts = []
                    s1 = "y = lambda : 1"
                    default_args = ast.parse(s1).body[0].value.args
                    ast.copy_location(default_args, predicate_)
                    for operand_ in ori_operands:
                        lambda_operand = ast.Lambda(args=default_args, body=operand_)
                        list_operand_elts.append(lambda_operand)
                    # ast.List(elts=list_operand_elts)
                    list_operand_ast = ast.List(elts=list_operand_elts)
                    ast.copy_location(list_operand_ast, predicate_)

                    s2 = "lambda x,y: x+y"
                    default_ops_args = ast.parse(s2).body[0].value.args
                    ast.copy_location(default_ops_args, predicate_)
                    for ops_ in ori_ops:
                        lambda_ops = ast.Lambda(args=default_ops_args,
                                                body=ast.Compare(left=ast.Name(id='x'), ops=[ops_],
                                                                 comparators=[ast.Name(id='y')]))
                        list_ops_elts.append(lambda_ops)
                    list_ops_ast = ast.List(elts=list_ops_elts)
                    ast.copy_location(list_ops_ast, predicate_)

                    s3 = "C.dict_con[(%d,%d,%d,%d)]" % (start_r, start_c, end_r, end_c)
                    d3 = ast.parse(s3).body[0].value
                    ast.copy_location(d3, predicate_)
                    new_predicate = ast.Call(func=ast.Name(id='help_visit_Compare'),
                                             args=[list_operand_ast, list_ops_ast, d3], keywords=[])
                    ast.copy_location(new_predicate, predicate_)
                    return new_predicate
                else:  # len(predicate_.ops) = 1
                    s2_ = "C.dict_con[(%d,%d,%d,%d)]" % (start_r, start_c, end_r, end_c)
                    s3_ = "help_modify_condition(b>0," + "C.dict_con[(%d,%d,%d,%d)])" % (start_r, start_c, end_r, end_c)
                    d2 = ast.parse(s2_).body[0].value
                    d3 = ast.parse(s3_).body[0].value
                    new_pred_ = ast.Call(func=d3.func, args=[predicate_, d2], keywords=[])
                    ast.copy_location(new_pred_, predicate_)
                    return new_pred_
            else:
                # help_modify_condition(predicate, dict_):
                s2_ = "C.dict_con[(%d,%d,%d,%d)]" % (start_r, start_c, end_r, end_c)
                s3_ = "help_modify_condition(b>0," + "C.dict_con[(%d,%d,%d,%d)])" % (start_r, start_c, end_r, end_c)
                d2 = ast.parse(s2_).body[0].value
                d3 = ast.parse(s3_).body[0].value
                new_pred_ = ast.Call(func=d3.func, args=[predicate_, d2], keywords=[])
                ast.copy_location(new_pred_, predicate_)
                return new_pred_

        def generic_visit(self, node):
            ast.NodeTransformer.generic_visit(self, node)
            try:
                start_r, start_c, end_r, end_c = node.lineno, node.col_offset, node.end_lineno, node.end_col_offset

                if (start_r, start_c, end_r, end_c) in C.dict_con.keys():
                    new_node = ast.fix_missing_locations(self.rewrite_atomic_predicate(node))
                    ast.copy_location(new_node, node)
                    return new_node
                return node
            except:
                return node


    R_Predicate = Rewrite_predicate()


    class Rewrite_tree(ast.NodeTransformer):
        def track_enter(self, node):
            # ast.NodeVisitor.generic_visit(self, node)
            node = self.generic_visit(node)
            lineno_ = node.lineno
            col_offset_ = node.col_offset
            count_ = ast.parse("S.dict_enter[(%d,%d)] +=1" % (lineno_, col_offset_)).body[0]
            ast.copy_location(count_, node)

            n = ast.Num(n=1)
            ast.copy_location(n, node)
            if_node_ = ast.If(test=n, body=[count_, node], orelse=[])
            ast.copy_location(if_node_, node)
            return if_node_

        def track_branch(self, node):
            node = self.generic_visit(node)
            lineno_ = node.lineno
            col_offset_ = node.col_offset
            count_stmt = ast.parse("S.dict_enter[(%d,%d)] +=1" % (lineno_, col_offset_)).body[0]
            ast.copy_location(count_stmt, node)
            # or isinstance(node, ast.For) or isinstance(node, ast.While) or isinstance(node, ast.AsyncFor)

            true_branch_ = ast.parse("B.dict_if[(%d,%d)]['true']+=1" % (lineno_, col_offset_)).body[0]
            false_branch_ = ast.parse("B.dict_if[(%d,%d)]['false']+=1" % (lineno_, col_offset_)).body[0]
            if isinstance(node, ast.If):
                new_predicate = ast.fix_missing_locations(R_Predicate.visit(node.test))
                new_body = [true_branch_] + node.body
                new_orelse = [false_branch_] + node.orelse
                new_node = ast.If(test=new_predicate, body=new_body, orelse=new_orelse)
                ast.copy_location(new_node, node)

                n = ast.Num(n=1)
                ast.copy_location(n, node)
                if_node_ = ast.If(test=n, body=[count_stmt, new_node], orelse=[])
                ast.copy_location(if_node_, node)
                return if_node_
            elif isinstance(node, ast.For) or isinstance(node, ast.AsyncFor):
                new_body = [true_branch_] + node.body
                new_orelse = [false_branch_] + node.orelse
                new_node = ast.For(target=node.target, iter=node.iter, body=new_body, orelse=new_orelse,
                                   type_comment=node.type_comment)
                ast.copy_location(new_node,
                                  node)  # the keypoint to overcome error 'For' object has no attribute 'lineno'

                n = ast.Num(n=1)
                ast.copy_location(n, node)
                if_node_ = ast.If(test=n, body=[count_stmt, new_node], orelse=[])
                ast.copy_location(if_node_, node)
                return if_node_
            elif isinstance(node, ast.While):
                new_predicate = ast.fix_missing_locations(R_Predicate.visit(node.test))
                new_body = [true_branch_] + node.body
                new_orelse = [false_branch_] + node.orelse
                new_node = ast.While(test=new_predicate, body=new_body, orelse=new_orelse)
                ast.copy_location(new_node, node)

                n = ast.Num(n=1)
                ast.copy_location(n, node)
                if_node_ = ast.If(test=n, body=[count_stmt, new_node], orelse=[])
                ast.copy_location(if_node_, node)
                return if_node_
            elif isinstance(node, ast.IfExp):
                current_Ifexp_test = node.test  # b>0
                new_predicate = ast.fix_missing_locations(R_Predicate.visit(node.test))
                s2_ = "B.dict_if[(%d,%d)]" % (lineno_, col_offset_)
                s3_ = "help_Ifexp(b>0," + "B.dict_if[(%d,%d)])" % (lineno_, col_offset_)
                d2 = ast.parse(s2_).body[0].value
                d3 = ast.parse(s3_).body[0].value

                new_IfExp_test = ast.Call(func=d3.func, args=[new_predicate, d2], keywords=[])
                ast.copy_location(new_IfExp_test, node.test)

                new_node = ast.IfExp(test=new_IfExp_test, body=node.body, orelse=node.orelse)
                ast.copy_location(new_node, node)
                return new_node
            elif isinstance(node, ast.ListComp):
                old_generator_list = node.generators  # a list of generators
                new_generator_list = []
                for generator_ in old_generator_list:
                    old_ifs_list = generator_.ifs  # it is a list of ifs
                    new_ifs_list = []
                    for old_ifs in old_ifs_list:
                        new_predicate = ast.fix_missing_locations(R_Predicate.visit(old_ifs))
                        ifs_lineno = old_ifs.lineno
                        ifs_col_offset = old_ifs.col_offset

                        s2_ = "B.dict_if[(%d,%d)]" % (ifs_lineno, ifs_col_offset)
                        d2 = ast.parse(s2_).body[0].value
                        s3_ = "help_Ifexp(b>0," + "B.dict_if[(%d,%d)])" % (ifs_lineno, ifs_col_offset)
                        d3 = ast.parse(s3_).body[0].value

                        new_ifs = ast.Call(func=d3.func, args=[new_predicate, d2], keywords=[])
                        ast.copy_location(new_ifs, old_ifs)
                        new_ifs_list.append(new_ifs)
                    new_generator_ = ast.comprehension(target=generator_.target, iter=generator_.iter, ifs=new_ifs_list,
                                                       is_async=generator_.is_async)
                    new_generator_list.append(new_generator_)
                new_node = ast.ListComp(elt=node.elt, generators=new_generator_list)
                ast.copy_location(new_node, node)
                return new_node

            elif isinstance(node,
                            ast.Try):  # B.dict_if[(lineno_handler, col_offset_handler)] = {'executed': 0, 'verbose': verbose_}
                list_old_handlers = node.handlers
                list_new_handlers = []
                for old_handler in list_old_handlers:
                    lineno_handler = old_handler.lineno
                    col_offset_handler = old_handler.col_offset
                    executed_check = \
                        ast.parse("B.dict_if[(%d,%d)]['executed'] +=1" % (lineno_handler, col_offset_handler)).body[0]
                    new_body_handler = [executed_check] + old_handler.body
                    new_handler = ast.ExceptHandler(type=old_handler.type, name=old_handler.name, body=new_body_handler)
                    list_new_handlers.append(new_handler)
                new_node = ast.Try(body=node.body, handlers=list_new_handlers, orelse=node.orelse,
                                   finalbody=node.finalbody)
                ast.copy_location(new_node, node)

                n = ast.Num(n=1)
                ast.copy_location(n, node)
                if_node_ = ast.If(test=n, body=[count_stmt, new_node], orelse=[])
                ast.copy_location(if_node_, node)
                return if_node_
            return node

        ### for branch coverage:
        visit_If = track_branch
        visit_For = track_branch
        visit_AsyncFor = track_branch
        visit_While = track_branch
        visit_Try = track_branch
        visit_IfExp = track_branch
        visit_ListComp = track_branch

        ### for statement coverage
        visit_FunctionDef = track_enter
        visit_AsyncFunctionDef = track_enter
        visit_ClassDef = track_enter
        visit_Return = track_enter
        visit_Delete = track_enter
        visit_Assign = track_enter
        visit_AugAssign = track_enter
        visit_AnnAssign = track_enter
        visit_With = track_enter
        visit_AsyncWith = track_enter
        visit_Raise = track_enter
        # Global
        visit_Assert = track_enter
        visit_Import = track_enter
        visit_ImportFrom = track_enter
        visit_Global = track_enter
        visit_Nonlocal = track_enter
        visit_Expr = track_enter
        visit_Pass = track_enter
        visit_Break = track_enter
        visit_Continue = track_enter


    R = Rewrite_tree()
    root = R.visit(root)
    sys.argv = [target] + args.remaining

    print("=====================================")
    print("Program Output")
    print("=====================================")

    # execute the instrumented target script
    exec(ast.unparse(root), {"S": S, "B": B, "help_Ifexp": help_Ifexp, "C": C, "help_visit_Compare": help_visit_Compare,
                             "help_modify_condition": help_modify_condition})

    print("=====================================")

    # print coverage percentage info
    num_stmt_covered = 0
    for i in S.dict_enter.values():
        if i > 0: num_stmt_covered += 1
    total_num_stmt = len(list(S.dict_enter.keys()))
    percent_stmt_covered = 100 * num_stmt_covered / total_num_stmt if total_num_stmt > 0 else 0
    print("Statement Coverage: " + "{:.2f}%".format(percent_stmt_covered) + " (%d / %d)" % (
    num_stmt_covered, total_num_stmt))
    total_num_branch = 0
    for coordinate in B.dict_if.keys():
        # count on ast.Try first
        if 'executed' in B.dict_if[coordinate]:
            total_num_branch += 1
        else:
            total_num_branch += 2
    num_branch_covered = 0
    for coordinate in B.dict_if.keys():
        # count on ast.Try first
        if 'executed' in B.dict_if[coordinate].keys():
            if B.dict_if[coordinate]['executed'] > 0:
                num_branch_covered += 1
        else:
            if B.dict_if[coordinate]['true'] > 0:
                num_branch_covered += 1
            if B.dict_if[coordinate]['false'] > 0:
                num_branch_covered += 1

    percent_branch_coverage = 100 * num_branch_covered / total_num_branch if total_num_branch > 0 else 0

    print("Branch Coverage: " + "{:.2f}%".format(percent_branch_coverage) + " (%d / %d)" % (
        num_branch_covered, total_num_branch))

    # compute condition coverage:
    total_num_condition = 0
    for coordinate in C.dict_con.keys():
        if C.dict_con[coordinate]['conjunction'] == True:
            total_num_condition += 2 * (len(C.dict_con[coordinate].keys()) - 1)
        else:
            total_num_condition += 2
    num_condition_covered = 0
    for coordinate in C.dict_con.keys():
        if C.dict_con[coordinate]['conjunction'] == True:
            for str_i in C.dict_con[coordinate].keys():
                if str_i != 'conjunction':
                    if C.dict_con[coordinate][str_i]['true'] > 0:
                        num_condition_covered += 1
                    if C.dict_con[coordinate][str_i]['false'] > 0:
                        num_condition_covered += 1
        else:
            if C.dict_con[coordinate]['true'] > 0:
                num_condition_covered += 1
            if C.dict_con[coordinate]['false'] > 0:
                num_condition_covered += 1
    percent_condition_covered = 100 * num_condition_covered / total_num_condition if total_num_condition > 0 else 0
    print("Condition Coverage: " + "{:.2f}%".format(percent_condition_covered) + " (%d / %d)" % (
    num_condition_covered, total_num_condition))

    # print verbose
    if args.verbose:
        print("=====================================")
        print("Covered Branches")
        # print verbose branch coverage
        list_branch_if = []
        all_coords = []
        for coordinate in B.dict_if.keys():
            all_coords.append(coordinate[0])
            all_coords.append(coordinate[1])
        max_coord = max(all_coords) + 1
        # # B.dict_if[(lineno_handler, col_offset_handler)] = {'executed': 0, 'verbose': verbose_}
        for coordinate in B.dict_if.keys():
            sp_index = max_coord * coordinate[0] + coordinate[1]
            if 'executed' in B.dict_if[coordinate].keys():
                list_branch_if.append((sp_index, coordinate[0], coordinate[1], B.dict_if[coordinate]['executed'],
                                       B.dict_if[coordinate]['verbose']))
            else:
                list_branch_if.append((sp_index, coordinate[0], coordinate[1], B.dict_if[coordinate]['true'],
                                       B.dict_if[coordinate]['false'], B.dict_if[coordinate]['verbose']))

        list_branch_if = sorted(list_branch_if, key=lambda x: x[0])
        for branch_ in list_branch_if:
            if len(branch_) == 6:
                if branch_[3] > 0:
                    print("Line %d: (" % branch_[1] + branch_[5] + ") ==> True")
                if branch_[4] > 0:
                    print("Line %d: (" % branch_[1] + branch_[5] + ") ==> False")
            elif len(branch_) == 5:  # ast.Try:
                if branch_[3] > 0:
                    print("Line %d: " % branch_[1] + branch_[4])

        print("=====================================")
        print("Covered Conditions")  # print verbose condition coverage
        con_all_coords = []
        for coordinate in C.dict_con.keys():
            con_all_coords.append(coordinate[0])
            con_all_coords.append(coordinate[1])
        if len(con_all_coords) > 0:
            max_con_coords = max(con_all_coords) + 1
        else:
            max_con_coords = 1

        max_conjunction_len = 0
        for coordinate in C.dict_con.keys():
            if C.dict_con[coordinate]['conjunction'] == True:
                max_conjunction_len = max(len(C.dict_con[coordinate].keys()) - 1, max_conjunction_len)
        max_conjunction_len += 2

        list_condition_ = []
        for coordinate in C.dict_con.keys():
            sp_con_index = max_con_coords * coordinate[0] + coordinate[1]
            sp_con_index = max_conjunction_len * sp_con_index
            if C.dict_con[coordinate]['conjunction'] == False:
                list_condition_.append((sp_con_index, coordinate[0], coordinate[1], C.dict_con[coordinate]['true'],
                                        C.dict_con[coordinate]['false'], C.dict_con[coordinate]['verbose']))
            else:
                for str_i in C.dict_con[coordinate].keys():
                    if str_i != 'conjunction':
                        sp_subcon_index = sp_con_index + int(str_i)
                        list_condition_.append((sp_subcon_index, coordinate[0], coordinate[1],
                                                C.dict_con[coordinate][str_i]['true'],
                                                C.dict_con[coordinate][str_i]['false'],
                                                C.dict_con[coordinate][str_i]['verbose']))
        list_condition_ = sorted(list_condition_, key=lambda x: x[0])
        for condition_ in list_condition_:
            if condition_[3] > 0:  # true branch
                print("Line %d: " % (condition_[1]) + condition_[-1] + " ==> True")
            if condition_[4] > 0:
                print("Line %d: " % (condition_[1]) + condition_[-1] + " ==> False")
    print("=====================================")