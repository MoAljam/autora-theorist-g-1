import numpy as np


def get_variable(variable_space, count=1):
    variables = []
    for i in range(0, count):
        variables = np.random.sample(variable_space)
    return variables