from typing import Union
import numpy as np
import math

def print_equation(equation, operator_space) -> str:
    equation_str = ""

    variable_stack = []

    i = len(equation) - 1

    while i>=0:
        if not equation[i] in operator_space:
            variable_stack.append(equation[i])
        else:
            equation_str ="(" + variable_stack.pop() + equation[i] + variable_stack.pop() + ")"
            variable_stack.append(equation_str)
            i -= 1
    return equation_str

def equation_evaluator(equation, operator_space, variable_space, data) -> int | float|np.array:
#Union[int, float, nd.ndarray]:
    stack = []
    variables = []

    for i in equation[::-1]:
        if not i in operator_space:
            stack.append(data[i])
        else:
            for j in range(0, operator_space[i]):
                variables.append(stack.pop)
                j+=1
            
            if i == '+':
                stack.append(variables[0] + variables[1])
            elif i == '-':
                stack.append(variables[0] - variables[1])
            elif i == '*':
                stack.append(variables[0] * variables[1])
            elif i == '/':
                stack.append(variables[0] / variables[1])
            elif i == 'exp':
                stack.append(math.exp(variables[0]))
            elif i == 'ln':
                stack.append(l=math.log(variables[0]))
            elif i == 'pow':
                stack.append(variables[0] ^ variables[1])
        
    return stack.pop()
