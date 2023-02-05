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
        self.__center = None  # Center point of the ellipse. When undefined equals None
        self.__radius = None  # Radius of the ball. When undefined equals None
        self.__acc = None  # The final training accuracy. When undefined equals None
        self.__success = False  # Boolean signifying if the training was successful

    # Define functions for accessing the private variables
    def center(self):
        assert self.__center is not None, "Node hasn't been trained"
        return self.__center

    def radius(self):
        assert self.__radius is not None, "Node hasn't been trained"
        return self.__radius

    def accuracy(self):
        assert self.__acc is not None, "Node hasn't been trained"
        return self.__acc

    def success(self):
        assert self.__success is not None, "Node hasn't been trained"
        return self.__success

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
            s.t. u_i - r^-2 * ||x_i - c||^2 >= 0      for i in 1, 2, ... , |X|
                 v_i + r^-2 * ||y_i - c||^2 - 2 >= 0  for i in 1, 2, ... , |Y|
                 u_i >= 0                             for i in 1, 2, ... , |X|
                 v_i >= 0                             for i in 1, 2, ... , |Y|
            where u_i and v_i are variables created from the relaxation of the original condition,
            c is the center point of the ball and gamma is the robustness factor
        :param X: The data points with the correct label. A numpy.ndarray of shape (m, d)
        :param Y: The data points with the incorrect labels. A numpy.ndarray of shape (n-m, d)
        :param verbose: Boolean telling whether the results of the optimization are printed
        :return: Void
        """
        c = self.center()
        m = X.shape[0]
        n = Y.shape[0]
        y_multiple = n / m
        gamma = (n + m) * self.base_gamma

        # Define the objective function
        def obj(params):
            return y_multiple * sum([params[i] for i in range(1, m + 1)]) + \
                   sum([params[i] for i in range(m + 1, m + n + 1)]) + \
                   gamma * params[0] ** 2

        cons = []
        # Define the constraints for x_i in X
        for i in range(0, m):
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: params[i + 1] - params[0] ** (-2) * norm(X[i] - c) ** 2})
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: params[i + 1]})

        # Define the constraints for y_i in Y
        for i in range(0, n):
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: params[i + 1 + m] + params[0] ** (-2) * norm(Y[i] - c) ** 2 - 2})
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: params[i + 1 + m]})

        cons.append({'type': 'ineq', 'fun': lambda params: params[0]})

        # Optimize for r
        variables = np.ones((1 + m + n,))
        res = minimize(obj, variables, method='trust-constr',
                       constraints=tuple(cons), tol=0.00001, options={'maxiter': 1000})

        if verbose:
            print(f"Results of optimizing for label: '{self.label.rstrip()}'")
            if res.success:
                print(f"Success: {res.success}")
            else:
                print(f"Success: {res.success}")
                print(f"Reason: {res.message}")
            print(f"Found optimum: {res.fun}")
            print(f"Found radius: {res.x[0]}")
            print(f"Other variables:\n{res.x[1:]}")
            print()

        u_avg = np.average(res.x[1:m + 1])
        v_avg = np.average(res.x[m + 1:1 + m + n])

        if u_avg > 1.8 or v_avg > 1.8:
            self.__success = False
        else:
            self.__success = True

        self.__radius = abs(float(res.x[0]))

        # Go through the points in X and Y and compute the accuracy
        self.__acc = (sum([1 for x in X if self.in_ball(x)]) + sum([1 for y in Y if not self.in_ball(y)])) / (m + n)

    def in_ball(self, point):
        """
        Checks if the inputted point falls within the ball defined in the node
        :param point: Input point. A numpy.ndarray of shape (d,)
        :return: Boolean
        """
        return self.__radius * norm(point - self.__center) ** 2 <= 1

    def dist(self, point):
        """
        Computes the distance from the surface of the ball to the given point
        :param point: Input point. A numpy.ndarray of shape (d,)
        :return: Floating point value signifying the distance
        """
        return norm(point - self.__center) - self.__radius
