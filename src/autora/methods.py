import numpy as np
import random

def node_addition(curr_equation, operator_space, variable_space):
    """
    input: array with current equation in prefix notation
    output: array with updated equation in prefix notation, i.e. with added node

    Example:
        # random seed
        >>> #random.seed(42)
        >>> #curr_equation = ['+', '2', '1']
        >>> #operator_space = {'+': 2, '-': 2, '*':2, '/':2, 'exp':1, 'ln':1, 'pow':2}
        >>> #variable_space = ['cons', 'eqn', 'X']
        >>> #node_addition(curr_equation, operator_space, variable_space)
        ['/', '2', '1']


    """

    #random.seed(42)
    replace_pos = np.random.randint(0, len(curr_equation))

    if curr_equation[replace_pos] in operator_space:
        temp_operator_space = operator_space.copy()
        del temp_operator_space[curr_equation[replace_pos]] # make sure operator does not get replace by same operator

        replace_node_options = [key for key, value in temp_operator_space.items() if value == operator_space.get(curr_equation[replace_pos])]
        replace_node = np.random.choice(replace_node_options)

        # ensure that there is no division by 0

    else:
        replace_node = np.random.choice(variable_space)

        ## add specific variable, constant or equation

    new_equation = curr_equation.copy()
    new_equation[replace_pos] = replace_node

    return new_equation