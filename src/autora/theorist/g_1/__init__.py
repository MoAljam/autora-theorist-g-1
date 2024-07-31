"""
Example Theorist
"""
from .utils import print_equation
from .methods import node_replacement, get_variable
from typing import Union

import numpy as np
import pandas as pd
import math

from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def equation_evaluator(equation, operator_space, variable_space, data):
    # return np.random.rand()
#Union[int, float, nd.ndarray]:
    stack = []

    # print("## equation: ", equation)
    # print("## operator_space: ", operator_space)
    # print("## variable_space: ", variable_space)
    # print("## data: ", data)
    for i in equation[::-1]:
        if not i in operator_space:
            if i == "eqn":
                stack.append(2)
            elif i == "cons":
                stack.append(np.random.randint(2,10))
            else:
                stack.append(data[i])
        else:
            variables = []
            # print("## stack: ", stack)
            for j in range(0, operator_space[i]):
                variables.append(stack.pop())
                j+=1
            # print("## variables: ", variables)
            # exit()
            if i == '+':
                stack.append((variables[0] + variables[1]))
            elif i == '-':
                stack.append((variables[0] - variables[1]))
            elif i == '*':
                stack.append((variables[0] * variables[1]))
            elif i == '/':
                stack.append((variables[0] / variables[1]))
            elif i == 'exp':
                stack.append(np.exp(variables[0]))
            elif i == 'ln':
                stack.append(np.log(variables[0]))
            elif i == 'pow':
                stack.append(np.power(variables[0] ,variables[1]))
    if not stack:
        return None
    return stack.pop()

class EquationFunction:
    def __init__(self, equation, idvs_names, operator_space, variable_space):
        self.equation = equation
        self.operator_space = operator_space
        self.variable_space = variable_space
        self.idvs_names = idvs_names

    def __call__(self, input_data):
        data = dict(zip(self.idvs_names, input_data))
        return equation_evaluator(self.equation, self.operator_space, self.variable_space, data)
    

def sample_equation(num_inputs: int=1):
    return lambda *x: np.random.rand()

def fit_measure(equation, condition, observation):
    if observation.ndim == 1:
        observation = observation[:, np.newaxis]
    if condition.ndim == 1:
        condition = condition[:, np.newaxis]
    # mean squared error
    print("## equation: ", equation)
    print("## condition: ", condition)
    print("## observation: ", observation)

    y = np.apply_along_axis(equation, 1, condition)
    return np.mean((y - observation) ** 2)

# draft
def random_equation(operator_space, variable_space, min_length=5, max_length=100):
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

class CustomMCMC(BaseEstimator):
    """
    Custom MCMC model
    """
    def __init__(self):
        # base equation of the model
        # define the operator space
        self.operator_space = {'+': 2, '-': 2, '*':2, '/':2, 'exp':1, 'ln':1, 'pow':2}
        #defined the variable space
        # cons: constant
        # eqn: is another equation, will cause a child fit process to run. 
        # self.temp_variable_space = ['cons', 'eqn']
        self.temp_variable_space = ['cons']


        # self.equation = lambda *args: None
        self.equation = None


    def fit(self, conditions: Union[pd.DataFrame, np.ndarray], observations: Union[pd.DataFrame, np.ndarray],
            max_iterations: int=100):
        #add independant variables to varable_space
        x = None
        if isinstance(conditions, np.ndarray) and conditions.ndim == 1:
            conditions = conditions[:, np.newaxis]

        if isinstance(conditions, pd.DataFrame):
            idvs_names = conditions.columns.tolist()
            x = conditions.values
        else:
            idvs_names = [f'x_{i}' for i in range(conditions.shape[1])]
            x = conditions

        y = None
        if isinstance(observations, np.ndarray) and observations.ndim == 1:
            observations = observations[:, np.newaxis]

        if isinstance(observations, pd.DataFrame):
            dvs_names = observations.columns.tolist()
            y = observations.values
        else:
            dvs_names = [f'x_{i}' for i in range(observations.shape[1])]
            y = observations

        self.variable_space = idvs_names + self.temp_variable_space

        rand_eqn = random_equation(self.operator_space, self.variable_space, min_length=5, max_length=10)
        rand_eqn = ['ln', "/",'cons', idvs_names[0]]
        eqn_old = EquationFunction(rand_eqn, idvs_names, self.operator_space, self.variable_space)
        fit_old = fit_measure(eqn_old, x, y)

        # basic MCMC algorithm
        for _ in range(max_iterations):
            # sample a new equation
            eqn = node_replacement(eqn_old.equation, self.operator_space, self.variable_space)
            eqn_new = EquationFunction(eqn, idvs_names, self.operator_space, self.variable_space)
            # eqn_new = sample_equation()
            # calculate the fit measure
            fit_new = fit_measure(eqn_new, x, y)
            # calculate the acceptance probability
            acceptance_prob = np.exp(fit_new - fit_old)

            # accept the new equation with the acceptance probability
            if np.random.rand() < acceptance_prob:
                eqn_old = eqn_new
                fit_old = fit_new

        self.equation = eqn_old

        return self
    
    def predict(self, conditions: Union[pd.DataFrame, np.ndarray]):
        if self.equation is None:
            raise ValueError("Model not fitted yet. Run the fit method first")
        
        if isinstance(conditions, np.ndarray) and conditions.ndim == 1:
            conditions = conditions[:, np.newaxis]
        if isinstance(conditions, pd.DataFrame):
            conditions = conditions.values

        y = np.apply_along_axis(self.equation, 1, conditions)
        # make sure the output is of shape (n_outputs, 1)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        return y

    def print_eqn(self):
        # equation = ['+', 'a','b']
        print("## equation: ", self.equation.equation)
        equation = print_equation(self.equation.equation, operator_space=self.operator_space)

        return equation