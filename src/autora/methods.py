import numpy as np


def get_variable(variable_space, count=1):
    variables = []
    for i in range(0, count):
        variables = np.random.sample(variable_space)
    return variables

def root_addition(curr_equation, operator_space, variable_space):

    
    # Randomly choose an operator to add
    new_operator = np.random.choice(list(operator_space.keys()))
    operator_type = operator_space[new_operator]
    
    # Add the new operator to the beginning of the equation
    new_equation = [new_operator] + curr_equation
    
    if operator_type == 2:
        # Add a new variable to the end of the equation if the operator arity is 2
        new_equation = new_equation+ get_variable(count=1, variable_space=variable_space)
       

    return new_equation
