from solver.solver import Solver

import ast
from ast import NodeTransformer, NodeVisitor, AST
import copy

class NodeCollector(NodeVisitor):
    """Collect all nodes in an AST."""
    def __init__(self) -> None:
        super().__init__()
        self._all_nodes= []

    def generic_visit(self, node: AST) -> None:
        self._all_nodes.append(node)
        return super().generic_visit(node)

    def collect(self, tree: AST):
        """Return a list of all nodes in tree."""
        self._all_nodes = []
        self.visit(tree)
        return self._all_nodes

def mark_nodes_level(node:AST, node_level):
    node._level = node_level
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, AST):
                    mark_nodes_level(item, node_level + 1)
        elif isinstance(value, AST):
            mark_nodes_level(value, node_level + 1)

class Collect_Nodes_level():
    def __init__(self):
        self.level_nodes = {}
    def traverse_visit(self, node: AST):
        node_level = node._level
        if node_level+1 not in self.level_nodes.keys():
            self.level_nodes[node_level+1] = []

        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        self.level_nodes[node_level+1].append(item)
                        self.traverse_visit(item)
            elif isinstance(value, AST):
                self.level_nodes[node_level+1].append(value)
                self.traverse_visit(value)

    def collect(self, tree: AST):
        self.level_nodes = {}
        self.level_nodes[0] = [tree]
        self.traverse_visit(tree)
        return self.level_nodes

class NodeMarker(NodeVisitor):
    def visit(self, node:AST):
        node.marked = True
        return super().generic_visit(node)

class ParentLoadMarker(NodeVisitor):
    def visit(self, node: AST):
        node.load_parent_marked = False  # False means remove
        return super().generic_visit(node)

class NodeReducer(NodeTransformer):
    def generic_visit(self, node: AST):
        ast.NodeTransformer.generic_visit(self, node)
        if not isinstance(node, ast.Load) and node.marked == False:
            if isinstance(node, ast.Module):
                return node
            else:
                return None
        else:
            return node

class RemoveLoadAtNode(NodeTransformer):
    def remove_load(self, node):
        for field, value in ast.iter_fields(node):
            if isinstance(value, AST):
                if isinstance(value, ast.Load):
                    delattr(node, field)
            elif isinstance(value, list):
                new_list = [ele for ele in value if not isinstance(ele, ast.Load)]
                value[:] = new_list

class TreeRemoveLoad(NodeTransformer):
    def __init__(self, level):
        self.level = level
    def generic_visit(self, node: AST):
        ast.NodeTransformer.generic_visit(self, node)
        try:
            if node._level == self.level - 1 and node.load_parent_marked == False:
                RemoveLoadAtNode().remove_load(node)
        except:
            pass
        return node

def TagNodes(input_tree:AST, level):
    mark_nodes_level(input_tree, 0)
    dict_node_level = Collect_Nodes_level().collect(input_tree)
    if level not in dict_node_level.keys():
        return []
    else:
        return dict_node_level[level]

def copy_and_reduce(tree:AST, reject_list) -> AST:
    NodeMarker().visit(tree)
    for node in reject_list:
        node.marked = False

    # copy tree and delete unmarked nodes
    new_tree = copy.deepcopy(tree)
    #NodeReducer().visit(new_tree)
    NR = NodeReducer()
    new_tree = NR.visit(new_tree)
    return new_tree

def copy_and_keep(tree:AST, keep_list, level) -> AST:
    NodeMarker().visit(tree)
    ParentLoadMarker().visit(tree)

    current_level_nodes_list = TagNodes(tree,level)
    #reject_nodes = [node_ for node_ in current_level_nodes_list if node_ not in keep_list]
    reject_nodes = [current_level_nodes_list[i] for i in range(len(current_level_nodes_list)) if i not in keep_list]
    for node in reject_nodes:
        node.marked = False

    keep_load_parent_list = []
    if level > 0:
        Level_relation_dict = level_relationship(TagNodes(tree, level-1))
        parent_tmp = []
        for keep_index in keep_list:
            if isinstance(current_level_nodes_list[keep_index], ast.Load):
                parent_tmp.append(Level_relation_dict[keep_index])
        parent_tmp = list(set(parent_tmp))
        keep_load_parent_list.extend(parent_tmp)
    for node_ in keep_load_parent_list:
        node_.load_parent_marked = True

    # copy tree and delete unmarked nodes
    new_tree = copy.deepcopy(tree)
    NR = NodeReducer()
    NR.visit(new_tree)

    # remove load
    TRM = TreeRemoveLoad(level)
    new_tree = TRM.visit(new_tree)

    return new_tree


def get_exception(code_sample):
    try:
        exec(code_sample, {})
        return None
    except Exception as e:
        return type(e)

def get_exception_tree(tree: AST):
    try:
        tree_str = ast.unparse(tree)
    except Exception as e:
        return type(e)
    else:
        try:
            exec(tree_str, {})
            return None
        except Exception as e:
            return type(e)

def prunned_error(tree, keep_list, level):
    tree = copy_and_keep(tree, keep_list=keep_list, level = level)
    return get_exception_tree(tree)

def collect_direct_child(node):
    l = []
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, AST):
                    l.append(item)
        elif isinstance(value, AST):
            l.append(value)
    return l
def collect_all_child_at_level(list_node):
    L = []
    for node_ in list_node:
        direct_child_nodes = collect_direct_child(node_)
        L.extend(direct_child_nodes)
    return L

def level_relationship(list_node):
    L = {}
    for node_ in list_node:
        direct_child_nodes = collect_direct_child(node_)
        L[node_] = len(direct_child_nodes)
    L_reverse = {}
    start_index = 0
    for node_ in L.keys():
        num_child = L[node_]
        end_index = start_index + num_child
        for i in range(start_index, end_index):
            L_reverse[i] = node_
            start_index = end_index
    return L_reverse

def ddmin(tree, must_keep_list, shrink_nodes_list, level, error_type):
    if len(shrink_nodes_list) == 1:
        return shrink_nodes_list
    n = len(shrink_nodes_list)
    left_shrink_list = shrink_nodes_list[:n//2]
    right_shrink_list = shrink_nodes_list[n//2 :]

    left_tree = copy_and_keep(tree, keep_list= must_keep_list + left_shrink_list,  level = level)
    right_tree = copy_and_keep(tree, keep_list=must_keep_list + right_shrink_list, level= level)
    if get_exception_tree(left_tree) == error_type:
        return ddmin(tree, must_keep_list, left_shrink_list, level, error_type)
    elif get_exception_tree(right_tree) == error_type:
        return ddmin(tree, must_keep_list, right_shrink_list, level, error_type)
    else:
        formal_node_list = ddmin(tree, must_keep_list + right_shrink_list, left_shrink_list, level, error_type) + ddmin(tree,must_keep_list + left_shrink_list, right_shrink_list, level, error_type)
        if prunned_error(tree, must_keep_list + formal_node_list, level) == error_type:
            return formal_node_list
        else:
            return shrink_nodes_list


class HierarchicalCodeSolver(Solver):
    def __init__(self):
        pass
    def solve(self, input_str):
        """
        :param input_str: a string of a code
        :return: a string of minized code
        """
        input_tree = ast.parse(input_str)
        target_error_type = get_exception(input_str)
        level = 0
        current_level_nodes_list = TagNodes(input_tree, level)

        while(len(current_level_nodes_list) != 0):
            current_level_nodes_indices = [i for i in range(len(current_level_nodes_list))]
            #minimizing_nodes_current_level_list = ddmin(input_tree, current_level_nodes_list, target_error_type)
            # ddmin(tree, must_keep_list, shrink_nodes_list, level, error_type)
            minimizing_node_indices_list = ddmin(input_tree, [], current_level_nodes_indices, level, target_error_type)

            reject_nodes_current_level_list = [current_level_nodes_list[i] for i in range(len(current_level_nodes_list)) if i not in minimizing_node_indices_list]

            # Prune
            NodeMarker().visit(input_tree)
            ParentLoadMarker().visit(input_tree)
            for node_ in reject_nodes_current_level_list:
                node_.marked = False

            keep_load_parent_list = []
            if level > 0:
                Level_relation_dict = level_relationship(TagNodes(input_tree, level - 1))
                parent_tmp = []
                for keep_index in minimizing_node_indices_list:
                    if isinstance(current_level_nodes_list[keep_index], ast.Load):
                        parent_tmp.append(Level_relation_dict[keep_index])
                parent_tmp = list(set(parent_tmp))
                keep_load_parent_list.extend(parent_tmp)
            for node_ in keep_load_parent_list:
                node_.load_parent_marked = True

            NR = NodeReducer()
            NR.visit(input_tree)
            # remove load
            TRM = TreeRemoveLoad(level)
            TRM.visit(input_tree)

            level = level + 1
            current_level_nodes_list = TagNodes(input_tree, level)
        return ast.unparse(input_tree)





