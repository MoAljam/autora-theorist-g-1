
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


