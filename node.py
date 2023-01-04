"""
Class defining an individual node and associated functions
"""
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize


class Node:

    def __init__(self, label, gamma=0.1):
        # The label of the node in question
        self.label = label

        # Parameter for balancing the need for robustness and correctness
        assert gamma > 0
        self.gamma = gamma

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

            min. sum(u_i for i in 1, 2, ... , |X|) + sum(v_i for i in 1, 2, ... , |Y|) + gamma * r
            s.t. u_i >= 0                      for i in 1, 2, ... , |X|
                 v_i >= 0                      for i in 1, 2, ... , |Y|

            where u_i >= 1 - r * ||x_i - v||^2, v_i >= r * ||y_i - v||^2 - 1, v is the center point
            of the ball and gamma is the robustness factor

        :param X: The data points with the correct label. A numpy.ndarray of shape (m, d)
        :param Y: The data points with the incorrect labels. A numpy.ndarray of shape (n-m, d)
        :param verbose: Boolean telling whether the results of the optimization are printed
        :return: Void
        """
        v = self.center()

        # Define the objective function
        def obj(r):
            return sum([1 - r * norm(x - v) ** 2 for x in X]) + \
                   sum([r * norm(y - v) ** 2 - 1 for y in Y]) + \
                   self.gamma * r

        cons = []
        # Define the constraints for x_i in X
        for x in X:
            cons.append({'type': 'ineq', 'fun': lambda r: 1 - r * norm(x - v) ** 2})

        # Define the constraints for y_i in Y
        for y in Y:
            cons.append({'type': 'ineq', 'fun': lambda r: r * norm(y - v) ** 2 - 1})

        # Optimize for r
        r0 = np.array(1.0)
        res = minimize(obj, r0, method='trust-constr', bounds=(0, None), constraints=cons)

        if verbose:
            print(res)

        self.__radius = r0[0]

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
