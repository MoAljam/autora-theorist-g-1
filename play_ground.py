from src.autora.theorist.g_1 import CustomMCMC, utils, methods
import numpy as np

def random_equation(operator_space, variable_space, min_length=5, max_length=100):
    # generate a randome equation with prefix notation
    # ensure it is a valid equation
    root = np.random.choice(list(operator_space.keys()))
    equation = [root]
    open_counter = operator_space[root]

    # create a random equation up to max length of 100
    # equation can be longer than max length
    while len(equation) < max_length:
        # print("## eq: ", equation)
        # print("## open_counter: ", open_counter)
        if open_counter == 0:
            if len(equation) > min_length:
                break
            else:
                equation.append(np.random.choice(list(operator_space.keys())))
                open_counter += operator_space[equation[-1]]
                continue

        if np.random.rand() > 0.5:
            equation.append(np.random.choice(list(operator_space.keys())))
            open_counter += operator_space[equation[-1]] - 1
        else:
            equation.append(np.random.choice(variable_space))
            open_counter -= 1
    
    # print("-"*30)
    # print("-"*30)
    # print("## intermidiate eq: ", equation)
    # print("## intermidiate open_counter: ", open_counter)
    # fill the rest of the equation with variables
    for _ in range(open_counter):
        equation.append(np.random.choice(variable_space))
        # print("## eq: ", equation)
        # print("## open_counter: ", open_counter)
    
    # print("-"*30)
    # print("-"*30)
    # print("## final eq: ", equation)
    # print("## final open_counter: ", open_counter)

    return equation
    
operator_space = {'+': 2, '-': 2, '*':2, '/':2, 'exp':1, 'ln':1, 'pow':2}
variable_space = ['cons', 'eqn', 'X']

equation = ["+", "2", "1"]

rep_eqn = methods.node_replacement(equation, operator_space, variable_space)

print(rep_eqn)

print(random_equation(operator_space, variable_space, min_length=5, max_length=10))