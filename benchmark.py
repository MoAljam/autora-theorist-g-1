# from autora.state import StandardState, on_state, Delta
from src.autora.theorist.g_1 import CustomMCMC

import autora.state as state
# experiment_runner
from autora.experiment_runner.synthetic.psychophysics.weber_fechner_law import weber_fechner_law
from autora.experiment_runner.synthetic.psychophysics.stevens_power_law import stevens_power_law
from autora.experiment_runner.synthetic.economics.expected_value_theory import expected_value_theory
# experimentalist
from autora.experimentalist.grid import grid_pool
from autora.experimentalist.random import random_pool, random_sample
# data handling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def benchmark(experiment_runner, theorist):

    # generate all conditions
    conditions = experiment_runner.domain()

    # generate all corresponding observations
    experiment_data = experiment_runner.run(conditions, added_noise=0.01)

    # get the name of the independent and independent variables
    ivs = [iv.name for iv in experiment_runner.variables.independent_variables]
    dvs = [dv.name for dv in experiment_runner.variables.dependent_variables]

    # extract the dependent variable (observations) from experiment data
    conditions = experiment_data[ivs]
    observations = experiment_data[dvs]

    # split into train and test datasets
    conditions_train, conditions_test, observations_train, observations_test = train_test_split(conditions, observations)

    # print("#### EXPERIMENT CONDITIONS (X):")
    # print(conditions)
    # print("#### EXPERIMENT OBSERVATIONS (Y):")
    # print(observations)

    # fit theorist
    theorist.fit(conditions_train, observations_train)

    # compute prediction for validation set
    predictions = theorist.predict(conditions_test)

    print("#### PREDICTIONS:")
    print(predictions)
    print("#### pridections shape: ", predictions.shape)

    # evaluate theorist performance
    error = np.power(predictions - observations_test.values, 2)
    error = np.mean(error)

    #check if the theorist has a print_eqn method
    if hasattr(theorist, 'print_eqn'):
        print("#### IDENTIFIED EQUATION:")
        print(theorist.print_eqn())

    print("#### VALIDATION SET MSE:")
    print(error)

    experiment_runner.plotter(model=theorist)
    plt.show()

class PolynomialRegressor:
    """
    This theorist fits a polynomial function to the data.
    """

    def __init__(self, degree: int = 3):
      self.poly = PolynomialFeatures(degree=degree, include_bias=False)
      self.model = LinearRegression()

    def fit(self, x, y):
      features = self.poly.fit_transform(x, y)
      self.model.fit(features, y)
      return self

    def predict(self, x):
      features = self.poly.fit_transform(x)
      return self.model.predict(features)

    def print_eqn(self):
        # Extract the coefficients and intercept
        coeffs = self.model.coef_
        intercept = self.model.intercept_

        # Handle multi-output case by iterating over each output's coefficients and intercept
        if coeffs.ndim > 1:
            for idx in range(coeffs.shape[0]):
                equation = f"y{idx+1} = {intercept[idx]:.3f}"
                feature_names = self.poly.get_feature_names_out()
                for coef, feature in zip(coeffs[idx], feature_names):
                    equation += f" + ({coef:.3f}) * {feature}"
                print(equation)
        else:
            equation = f"y = {intercept:.3f}"
            feature_names = self.poly.get_feature_names_out()
            for coef, feature in zip(coeffs, feature_names):
                equation += f" + ({coef:.3f}) * {feature}"
            print(equation)


if __name__ == "__main__":

    my_theorist = CustomMCMC()
    # my_theorist = PolynomialRegressor()

    # run benchmark
    benchmark(experiment_runner = stevens_power_law(), theorist = my_theorist)
    # run benchmark
    # benchmark(experiment_runner = weber_fechner_law(), theorist = my_theorist)
    # # run benchmark
    # benchmark(experiment_runner = expected_value_theory(), theorist = my_theorist)
