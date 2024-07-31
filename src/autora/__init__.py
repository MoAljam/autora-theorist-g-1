"""
Example Theorist
"""
from .utils import print_equation, equation_evaluator, random_equation
from .methods import node_replacement, get_variable, root_addition
from typing import Union

import numpy as np
import pandas as pd
import math

from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



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
            if np.random.rand() > 0.5:
                eqn = node_replacement(eqn_old.equation, self.operator_space, self.variable_space)
            else:
                eqn = root_addition(eqn_old.equation, self.operator_space, self.variable_space)
            # eqn = node_replacement(eqn_old.equation, self.operator_space, self.variable_space)
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