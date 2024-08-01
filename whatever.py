import numpy as np

def build_expression(prefix_list):
    """
    Converts a prefix notation list to a Python expression string.
    """
    stack = []

    # Iterate over the elements in reverse order
    for token in reversed(prefix_list):
        if token in ['+', '-', '*', '/']:
            # Pop two operands for binary operators
            operand1 = stack.pop()
            operand2 = stack.pop()
            stack.append(f"({operand1} {token} {operand2})")
        elif token in ['pow', 'exp', 'ln']:
            # Pop the appropriate number of operands for specific functions
            if token == 'pow':
                base = stack.pop()
                exponent = stack.pop()
                stack.append(f"np.{token}({base}, {exponent})")
            else:
                operand = stack.pop()
                stack.append(f"np.{token}({operand})")
        else:
            # For numbers and variables directly push to stack
            stack.append(token)

    # The last element on the stack is the full expression
    return stack[0]

def create_function(prefix_list):
    """
    Creates a function from a prefix notation list.
    """
    expr = build_expression(prefix_list)
    # Create a function dynamically with numpy and the built expression
    func_str = f"""
def generated_func(x_1, x_2):
    return {expr}
"""
    # Create a local namespace and execute function definition string
    local_namespace = {}
    exec(func_str, {'np': np}, local_namespace)
    return local_namespace['generated_func']

# Example usage
prefix_eq = ['+', '+', 'pow', '10', 'x_1', '2', 'x_2']
func = create_function(prefix_eq)
print(func)
print(func(3, 4))  # Example evaluation
