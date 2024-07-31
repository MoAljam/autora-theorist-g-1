"""
Example Theorist
"""
from typing import Union
import utils, methods

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



class ExampleRegressor(BaseEstimator):
    """
    Include inline mathematics in docstring \\(x < 1\\) or $c = 3$
    or block mathematics:

    \\[
        x + 1 = 3
    \\]


    $$
    y + 1 = 4
    $$

    """

    def __init__(self, degree: int = 3):
      # define the operator space
      self.operator_space = {'+': 2, '-': 2, '*':2, '/':2, 'exp':1, 'ln':1, 'pow':2}
      #defined the variable space
      # cons: constant
      # eqn: is another equation, will cause a child fit process to run. 
      self.temp_variable_space = ['cons', 'eqn']

    def fit(self, x, y):
      #add independant variables to varable_space
      self.variable_space = x + self.temp_variable_space

      features = self.poly.fit_transform(x, y)
      self.model.fit(features, y)
      return self

    def predict(self, x):
      features = self.poly.fit_transform(x)
      return self.model.predict(features)

    def print_eqn(self):
        equation = ['+', 'a','b']
        utils.print_equation(equation, operator_space=self.operator_space)