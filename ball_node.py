"""
Class defining an individual node for ball classifier and associated functions
"""
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize


class BallNode:

    def __init__(self, label, base_gamma=1):
        # The label of the node in question
        self.label = label

        # Parameter for balancing the need for robustness and correctness
        assert base_gamma > 0
        self.base_gamma = base_gamma

        # Define private variables
        self.__center = None  # Center point of the ball. When undefined equals None
        self.__radius = None  # Radius of the ball. When undefined equals None
        self.__acc = None  # The final training accuracy. When undefined equals None

    # Define functions for accessing the private variables
    def center(self):
        return self.__center

    def radius(self):
        return self.__radius

    def accuracy(self):
        return self.__acc

    # Other functions
    def find_center(self, X):
        """
        Find the center point of the sphere. This is just the arithmetic mean of the data points
        :param X: The data points with the correct label. A numpy.ndarray of shape (m, d)
        :return: Void
        """
        self.__center = np.mean(X, axis=0)

    def find_radius(self, X, Y, verbose=True):
        """
        Optimize for the radius - that is solve the constrained nonlinear optimization problem

            min. sum(u_i for i in 1, 2, ... , |X|) + sum(v_i for i in 1, 2, ... , |Y|) + gamma * r ^ 2
            s.t. u_i - r * ||x_i - c||^2 >= 0      for i in 1, 2, ... , |X|
                 v_i + r * ||y_i - c||^2 - 2 >= 0  for i in 1, 2, ... , |Y|
                 u_i >= 0                          for i in 1, 2, ... , |X|
                 v_i >= 0                          for i in 1, 2, ... , |Y|

            where u_i and v_i are variables created from the relaxation of the original condition,
            c is the center point of the ball and gamma is the robustness factor

        :param X: The data points with the correct label. A numpy.ndarray of shape (m, d)
        :param Y: The data points with the incorrect labels. A numpy.ndarray of shape (n-m, d)
        :param verbose: Boolean telling whether the results of the optimization are printed
        :return: Void
        """
        v = self.center()
        m = X.shape[0]
        n = Y.shape[0]
        gamma = (n + m) * self.base_gamma

        # Define the objective function
        def obj(params):
            return sum([params[i] for i in range(1, m + 1)]) + \
                   sum([params[i] for i in range(m + 1, m + n + 1)]) + \
                   gamma * params[0] ** 2

        cons = []
        # Define the constraints for x_i in X
        for i in range(0, m):
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: params[i + 1] - params[0] * norm(X[i] - v) ** 2})
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: params[i + 1]})

        # Define the constraints for y_i in Y
        for i in range(0, n):
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: params[i + 1 + m] + params[0] * norm(Y[i] - v) ** 2 - 2})
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: params[i + 1 + m]})

        cons.append({'type': 'ineq', 'fun': lambda params: params[0]})

        # Optimize for r
        variables = np.ones((1 + m + n,))
        res = minimize(obj, variables, method='SLSQP', constraints=tuple(cons))

        if verbose:
            print(res)
            print()

        # print(f"\nActual solution: {obj(res.x)}\n")

        self.__radius = float(res.x[0])

        # Go through the points in X and compute the accuracy
        self.__acc = sum([1 for x in X if self.in_ball(x)]) / X.shape[0]

    def in_ball(self, point):
        """
        Checks if the inputted point falls within the ball defined in the node
        :param point: Input point. A numpy.ndarray of shape (d,)
        :return: Boolean
        """
        return self.__radius * norm(point - self.__center) ** 2 <= 1

    def in_ball_with_dist(self, point):
        """
        Checks if the inputted point falls within the ball defined in the node
        and returns the distance to the center (as defined by l_2 norm) if that
        is the case and -1 otherwise
        :param point: Input point. A numpy.ndarray of shape (d,)
        :return:
        """
        return norm(point - self.__center) if self.in_ball(point) else -1.0
