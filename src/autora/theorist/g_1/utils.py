import numpy as np
from numbers import Number

np.seterr(all='raise')

RANGE_CONSTANTS = (0.5, 3)
RANGE_EQN = RANGE_CONSTANTS
def replace_cons_eqn(equation):
    for i in range(0, len(equation)):
        # if equation[i] == 'eqn':
        # #     equation[i] = random_equation(equation, equation, min_length=3, max_length=10)
        # elif equation[i] == 'cons':
        if equation[i] == 'cons':
            equation[i] = str(np.random.uniform(*RANGE_CONSTANTS))
    return equation

def equation_evaluator(equation, operator_space, variable_space, data):
    """
    Evaluates an equation in prefix notation with error handling for edge cases.
    
    Parameters:
    - equation (list): The equation in prefix notation.
    - operator_space (dict): A dictionary mapping operators to the number of operands they take.
    - variable_space (dict): A dictionary mapping symbolic variable names to values.
    - data (dict): A dictionary mapping variable identifiers in the equation to their values.

    Returns:
    - The result of evaluating the equation or None if an error occurs.
    """
    stack = []
    i = len(equation) - 1
    while i >= 0:
        token = equation[i]
        if token in operator_space and operator_space[token] > 0:
            # Ensure there are enough operands in the stack for the operation
            if len(stack) < operator_space[token]:
                return None  # Not enough operands for the operation
            
            operands = [stack.pop() for _ in range(operator_space[token])]
            
            try:
                if token == '+':
                    result = np.add(*operands)
                elif token == '-':
                    result = np.subtract(*operands)
                elif token == '*':
                    result = np.multiply(*operands)
                elif token == '/':
                    if operands[1] == 0:
                        return np.NAN  # Avoid division by zero
                    result = np.divide(*operands)
                elif token == 'exp':
                    # bound input to avoid overflow
                    exponent = np.clip(operands[0], -10, 10)
                    result = np.exp(exponent)
                elif token == 'ln':
                    if operands[0] <= 0:
                        return np.NAN  # Log of non-positive number is not valid
                    result = np.log(operands[0],)
                elif token == 'pow':
                    if operands[0] == 0 and operands[1] <= 0:
                        return np.NAN  # 0^0 or 0 to a negative power is not valid
                    exponent = np.clip(operands[1], -10, 10)
                    result = np.power(operands[0], exponent)
            except FloatingPointError as e:
                print("#### Error during computation:", str(e))
                print("## operands: ", operands)
                print("## token: ", token)
                print("## stack: ", stack)
                print("## equation: ", equation)
                return np.NAN  # Handle mathematical overflows
            stack.append(result)
        else:
            # Handle variables and constants
            if token.replace('.','',1).isdigit():
                stack.append(float(token))  # Assign a random constant if 'cons'
            # elif token == "eqn":
            #     stack.append(*RANGE_EQN)  # Example value for 'eqn', adjust as needed
            else:
                stack.append(data[token])  # Use the value from the data dictionary
        i -= 1

    return stack.pop() if stack else None

def random_equation(operator_space, variable_space, min_length=5, max_length=10):
    # generate a randome equation with prefix notation
    # max length can be exceeded
    root = np.random.choice(list(operator_space.keys()))
    equation = [root]
    open_counter = operator_space[root]

    # create a random equation up to max length of 100
    # equation can be longer than max length
    while len(equation) < max_length:
        # print("## eq: ", equation, "### open_counter: ", open_counter)
        if open_counter == 0:
            if len(equation) > min_length:
                break
            else:
                new_operator = np.random.choice(list(operator_space.keys()))
                equation.insert(0, new_operator)
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
    
    # for _ in range(open_counter):
    #     equation.append(np.random.choice(variable_space))
    # prune the equation until the open_counter is zero (valid equation)
    while open_counter > 0:
        if len(equation) == 1:
            break
        if equation[-1] in operator_space:
            open_counter = open_counter - operator_space[equation[-1]] + 1
            del equation[-1]
        else: # symbol is a variable
            open_counter += 1
            del equation[-1]

    # case where equation is only one operator
    for _ in range(open_counter):
        equation.append(np.random.choice(variable_space))
        open_counter -= 1

        # print("## eq: ", equation, "### open_counter: ", open_counter)
    
    # print("-"*30)
    # print("## final eq: ", equation,"### final open_counter: ", open_counter)
    equation = replace_cons_eqn(equation)

    return equation

# def print_equation(equation, operator_space) -> str:
#     equation_str = ""

#     variable_stack = []

#     i = len(equation) - 1

#     while i>=0:
#         if not equation[i] in operator_space:
#             variable_stack.append(equation[i])
#         else:
#             # print("## variable_stack: ", variable_stack)
#             if operator_space[equation[i]] == 1:
#                 equation_str ="(" + equation[i] + "("+variable_stack.pop()+")" + ")"
#             else:
#                 equation_str ="(" + variable_stack.pop() + equation[i] + variable_stack.pop() + ")"
#             variable_stack.append(equation_str)
#         i -=1
#     return equation_str

def print_equation(equation, operator_space):
    """
    Converts a prefix notation list back to a readable infix equation string.

    Parameters:
    - equation (list): The equation in prefix notation.
    - operator_space (dict): A dictionary mapping operators to the number of operands they take.

    Returns:
    - str: A string representation of the equation in infix format.
    """
    stack = []
    # Process the equation from right to left
    for token in reversed(equation):
        if token in operator_space:
            if operator_space[token] == 1:
                # Unary operator, needs one operand
                operand = stack.pop() if stack else 'undefined'
                stack.append(f"({token}({operand}))")
            else:
                # Binary operator, needs two operands
                operand1 = stack.pop() if stack else 'undefined'
                operand2 = stack.pop() if stack else 'undefined'
                stack.append(f"({operand1} {token} {operand2})")
        else:
            # It's a variable or a number, push directly onto the stack
            stack.append(token)

    # The final item on the stack is the fully formatted equation
    return stack.pop() if stack else ''

# Example usage
if __name__ == "__main__":
    operator_space = {'+': 2, '-': 2, '*': 2, '/': 2, 'exp': 1, 'ln': 1, 'pow': 2}
    prefix_eq = ['+', 'exp', 'X', 'pow', 'Y', '2']
    equation_str = print_equation(prefix_eq, operator_space)
    print("Formatted Equation:", equation_str)

    # random equation generation
    variable_space = ['X', 'Y', 'cons']
    equation = random_equation(operator_space, variable_space)
    print("Random Equation:", equation)
