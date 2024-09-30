"""
Example Theorist
"""
from .utils import print_equation, equation_evaluator, random_equation
from .methods import root_addition, root_removal, node_replacement, get_variable
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
    def __str__(self):
        return print_equation(self.equation, operator_space=self.operator_space)
    

def sample_equation_raw(curr_equation, operator_space, variable_space):
    tree_modification_methods = [root_addition, root_removal, node_replacement]

    # method = np.random.choice(tree_modification_methods)
    # give higher probability to node_removel
    method = np.random.choice(tree_modification_methods , p=[0.55, 0.05, 0.4])
    print("## method: ", method.__name__)
    eqn = method(curr_equation, operator_space, variable_space)
    return eqn

def fit_measure(equation, condition, observation):
    if observation.ndim == 1:
        observation = observation[:, np.newaxis]
    if condition.ndim == 1:
        condition = condition[:, np.newaxis]
    # mean squared error
    print("## equation: ", equation.equation)
    # print("## condition: ", condition)
    # print("## observation: ", observation)

    y = np.apply_along_axis(equation, 1, condition)
    # print("## y: ", y)
    try:
        y = np.mean((y - observation) ** 2)
    except:
        print("## error in fit_measure")
        y = np.inf
    y = -y
    return y ## leaving this for moh to fix.XD 


class CustomMCMC(BaseEstimator):
    """
    Custom MCMC model
    """
    def __init__(self, max_eqn_length=20, max_iterations=100):
        # base equation of the model
        # define the operator space
        self.max_eqn_length = max_eqn_length
        self.max_iterations = max_iterations
        self.operator_space = {'+': 2, '-': 2, '*':2, '/':2, 'exp':1, 'ln':1, 'pow':2}
        # self.operator_space = {'+': 2, '-': 2, '*':2, '/':2, 'exp':1, 'ln':1, 'pow':2, 'cons':0}

        # defined the variable space
        # cons: constant
        # eqn: is another equation, will cause a child fit process to run. 
        # self.temp_variable_space = ['cons', 'eqn']
        self.temp_variable_space = ['cons']


        # self.equation = lambda *args: None
        self.model = None


    def fit(self, conditions: Union[pd.DataFrame, np.ndarray], observations: Union[pd.DataFrame, np.ndarray]):
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

        # Danger
        # Danger
        self.variable_space = idvs_names + self.temp_variable_space ## DO NOT TOUCHT, VERY CRITICAL INTERNAL
        # Danger
        # Danger

        rand_eqn_raw = random_equation(self.operator_space, self.variable_space, min_length=2, max_length=5)
        # rand_eqn = ['ln', "/",'cons', idvs_names[0]]
        eqn_old = EquationFunction(rand_eqn_raw, idvs_names, self.operator_space, self.variable_space)

        fit_old = fit_measure(eqn_old, x, y)

        fit_values = [fit_old]
        # basic MCMC algorithm
        for i in range(self.max_iterations):
            print("##--------------------- iteration: ", i)
            len_eqn = len(eqn_old.equation)
            print("## len_eqn: ", len_eqn)
            print("## eqn_old: ", eqn_old.equation)
            if len_eqn > self.max_eqn_length:
                eqn_raw =root_removal(eqn_old.equation, self.operator_space, self.variable_space)
            else:
                # sample a new equation
                eqn_raw = sample_equation_raw(eqn_old.equation, self.operator_space, self.variable_space)
            # eqn = node_replacement(eqn_old.equation, self.operator_space, self.variable_space)
            eqn_new = EquationFunction(eqn_raw, idvs_names, self.operator_space, self.variable_space)
            # eqn_new = sample_equation()
            # calculate the fit measure
            fit_new = fit_measure(eqn_new, x, y)
            # calculate the acceptance probability
            fit_diff = fit_new - fit_old
            beta = 2
            if fit_diff > 0:
                acceptance_prob = 1.1 # always accept the better equation
            else:
                # acceptance_prob = 0.2
                try:
                    # acceptance_prob = np.exp(fit_diff * beta) # accept the worse equation with exp probability of the difference
                    acceptance_prob = 1/(1 + np.exp(-fit_diff * beta))
                except:
                    acceptance_prob = 0

            print("## fit_old: ", fit_old)
            print("## fit_new: ", fit_new)
            print("## acceptance_prob: ", acceptance_prob)
            # accept the new equation with the acceptance probability
            if np.random.rand() < acceptance_prob:
                eqn_old = eqn_new
                fit_old = fit_new
                fit_values.append(fit_new)
                print("## accepted")

        self.model = eqn_old
        print("## model's equation: ", eqn_old.equation)
        print("## model's fit: ", fit_old)

        fit_values = np.array(fit_values)
        # print("## fit_values: ", fit_values)
        # remove problematic values
        fit_values_plot = fit_values[~np.isnan(fit_values) & ~np.isinf(fit_values)]
        print("## fit_values_plot: ", fit_values)
        plt.plot(fit_values_plot)
        plt.title("Fit values")
        plt.show()

        return self
    
    def predict(self, conditions: Union[pd.DataFrame, np.ndarray]):
        if self.model is None:
            raise ValueError("Model not fitted yet. Run the fit method first")
        
        if isinstance(conditions, np.ndarray) and conditions.ndim == 1:
            conditions = conditions[:, np.newaxis]
        if isinstance(conditions, pd.DataFrame):
            conditions = conditions.values

        y = np.apply_along_axis(self.model, 1, conditions)
        # make sure the output is of shape (n_outputs, 1)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        return y

    def print_eqn(self):
        # equation = ['+', 'a','b']
        # print("## equation: ", self.model.equation)
        equation = print_equation(self.model.equation, operator_space=self.operator_space)

        return equation