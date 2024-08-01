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
            # print("## variable_stack: ", variable_stack)
            if operator_space[equation[i]] == 1:
                equation_str ="(" + equation[i] + "("+variable_stack.pop()+")" + ")"
            else:
                equation_str ="(" + variable_stack.pop() + equation[i] + variable_stack.pop() + ")"
            variable_stack.append(equation_str)
        i -=1
    return equation_str


def equation_evaluator(equation, operator_space, variable_space, data):
    # return np.random.rand()
#Union[int, float, nd.ndarray]:
    stack = []

    # print("## equation: ", equation)
    # print("## operator_space: ", operator_space)
    # print("## variable_space: ", variable_space)
    # print("## data: ", data)
    # print("##> ",type(equation), " > ", equation)
    i = len(equation) - 1
    while i>=0:
        if not equation[i] in operator_space:
            if equation[i] == "eqn":
                stack.append(2)
            elif equation[i] == "cons":
                stack.append(np.random.randint(2,10))
            else:
                stack.append(data[equation[i]])
        else:
            variables = []
            # print("## stack: ", stack)
            for j in range(0, operator_space[equation[i]]):
                variables.append(stack.pop())
                j+=1
            # print("## variables: ", variables)
            # exit()
            if equation[i] == '+':
                stack.append((variables[0] + variables[1]))
            elif equation[i] == '-':
                stack.append((variables[0] - variables[1]))
            elif equation[i] == '*':
                stack.append((variables[0] * variables[1]))
            elif equation[i] == '/':
                if not variables[1] == 0:
                    stack.append((variables[0] / variables[1]))
                else:
                    return(-math.inf)
            elif equation[i] == 'exp':
                stack.append(np.exp(variables[0]))
            elif equation[i] == 'ln':
                if not variables[0] == 0:
                    stack.append(np.log(variables[0]))
                else:
                    return(-math.inf)
            elif equation[i] == 'pow':
                stack.append(np.power(variables[0] ,variables[1]))
        i -=1
    if not stack:
        return None
    return stack.pop()

def random_equation(operator_space, variable_space, min_length=5, max_length=10):
    # generate a randome equation with prefix notation
    # max length can be exceeded
    root = np.random.choice(list(operator_space.keys()))
    equation = [root]
    open_counter = operator_space[root]

    # create a random equation up to max length of 100
    # equation can be longer than max length
    while len(equation) < max_length:
        # print("## eq: ", equation ### open_counter: ", open_counter)
        if open_counter == 0:
            if len(equation) > min_length:
                break
            else:
                equation.insert(0, np.random.choice(list(operator_space.keys())))
                open_counter += operator_space[equation[0]] - 1
                continue

        if np.random.rand() > 0.5:
            equation.append(np.random.choice(list(operator_space.keys())))
            open_counter += operator_space[equation[-1]] - 1
        else:
            equation.append(np.random.choice(variable_space))
            open_counter -= 1
    
    # print("-"*30)
    # print("## intermidiate eq: ", equation ### intermidiate open_counter: ", open_counter)
    # fill the rest of the equation with variables
    for _ in range(open_counter):
        equation.append(np.random.choice(variable_space))
        # print("## eq: ", equation ### open_counter: ", open_counter)
    
    # print("-"*30)
    # print("## final eq: ", equation ### final open_counter: ", open_counter)

    return equation