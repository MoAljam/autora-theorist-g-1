from .utils import random_equation, replace_cons_eqn
import numpy as np


def node_replacement(curr_equation, operator_space, variable_space):
    """
    input:
        curr_equation: array with current equation in prefix notation
        operator_space: space of operators
        variable_space: space of variables
    output:
    array with updated equation in prefix notation, i.e. with added node

    Example:
        >>> np.random.seed(42)
        >>> curr_equation = ['+', '2', '1']
        >>> operator_space = {'+': 2, '-': 2, '*':2, '/':2, 'exp':1, 'ln':1, 'pow':2}
        >>> variable_space = ['cons', 'eqn', '1', '2']
        >>> node_replacement(curr_equation, operator_space, variable_space)
        ['+', '2', '2']
    """

    # np.random.seed(42)
    replace_pos = np.random.randint(0, len(curr_equation))

    if curr_equation[replace_pos] in operator_space:
        temp_operator_space = operator_space.copy()
        del temp_operator_space[curr_equation[replace_pos]]  # make sure operator does not get replaced by same operator

        replace_node_options = [
            key for key, value in temp_operator_space.items() if value == operator_space.get(curr_equation[replace_pos])
        ]
        replace_node = np.random.choice(replace_node_options)

    else:
        replace_node = np.random.choice(variable_space)

        # add specific variable, constant or equation

    new_equation = curr_equation.copy()
    new_equation[replace_pos] = replace_node

    if ("cons" in new_equation) or ("eqs" in new_equation):
        replace_cons_eqn(new_equation)

    return new_equation


def get_variable(variable_space, count=1):

    variables = []
    for i in range(0, count):
        variables = np.random.choice(variable_space)
    print(variables)
    return variables


def root_addition(curr_equation, operator_space, variable_space):
    """ "
    Example:
        >>> np.random.seed(42)
        >>> curr_equation = ['+', '2', '1']
        >>> operator_space = {'+': 2, '-': 2, '*':2, '/':2, 'exp':1, 'ln':1, 'pow':2}
        >>> variable_space = ['cons', 'eqn', '1', '2']
        >>> root_addition(curr_equation, operator_space, variable_space)
        ['pow', '+', '2', '1', '2']
    """

    # np.random.seed(42)

    # Randomly choose an operator to add
    new_operator = np.random.choice(list(operator_space.keys()))
    operator_type = operator_space[new_operator]

    # Add the new operator to the beginning of the equation
    new_equation = [new_operator] + curr_equation.copy()

    if operator_type == 2:
        # Add a new variable to the end of the equation if the operator arity is 2
        new_equation = new_equation + [(np.random.choice(variable_space))]

    if ("cons" in new_equation) or ("eqs" in new_equation):
        replace_cons_eqn(new_equation)

    return new_equation


def root_removal(curr_equation, operator_space, variable_space):
    """
    Example:
        >>> np.random.seed(42)
        >>> curr_equation = ['+', '2', '1']
        >>> operator_space = {'+': 2, '-': 2, '*':2, '/':2, 'exp':1, 'ln':1, 'pow':2}
        >>> variable_space = ['cons', 'eqn', '1', '2']
        >>> root_removal(curr_equation, operator_space, variable_space)
        ['2']
    """

    # np.random.seed(42)
    new_equation = curr_equation.copy()
    # check number of arguments operator at root is expecting

    # CASE 1: root is a variable / constant (length of equation is 1)
    # if equation has only one element (variable or constant), return it as is (don't remove root)
    if len(new_equation) == 1:
        # print("## CASE 1: root is a variable / constant")
        return new_equation

    # CASE 2: root is an operator
    # CASE 2.1: operator at root expects only one argument

    # remove operator at root
    if operator_space.get(new_equation[0]) == 1:
        # print("## CASE 2.1: operator at root expects only one argument")
        del new_equation[0]
        return new_equation

    # CASE 2.2: operator at root expects two arguments

    # if operator at root expects only one argument, new equation is done
    # if operator at root expects two arguments, remove operator and right child
    # TO DO: figure out how to remove right child
    if operator_space.get(curr_equation[0]) == 2:
        # print("## CASE 2.2: operator at root expects two arguments")

        # get both children
        child_left = new_equation[1]
        # mark where right child left ends
        if child_left in operator_space:
            open_counter_left = operator_space.get(child_left)
        else:  # it's a variable
            open_counter_left = 0
        # assume right child is available and starts directly after left child
        idx_child_right = 2
        # keep pushing right until the end of the left node is reached (open_counter_left == 0)
        while open_counter_left > 0:
            current_node = new_equation[idx_child_right]
            if new_equation[idx_child_right] in operator_space:
                open_counter_left = open_counter_left + operator_space.get(current_node) - 1
            else:  # symbol is a variable
                open_counter_left -= 1
            idx_child_right += 1

        # # double check if right child is available or if the root has only one child
        # if idx_child_right < len(new_equation):
        #     child_right = new_equation[idx_child_right]
        # else:
        #     child_right = None

        del new_equation[0]  # remove root
        if np.random.rand() > 0.5:  # remove left child
            # indecies got shifted left by one (-1) after removing root
            del new_equation[0 : idx_child_right - 1]
        else:  # remove right child
            del new_equation[idx_child_right - 1 :]

        return new_equation

    else:  # root operator arity > 3
        raise ValueError("Operator (root) arity (>3) not supported")
