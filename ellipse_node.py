"""
Class defining an individual node for ellipse classifier and associated functions
"""
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize


class EllipseNode:

    def __init__(self, label, base_gamma=1):
        # The label of the node in question
        self.label = label

        # Parameter for balancing the need for robustness and correctness
        assert base_gamma > 0
        self.base_gamma = base_gamma

        # Define private variables
        self.__center = None  # Center point of the ellipse. When undefined equals None
        self.__matrix = None  # Characteristic matrix of the ellipse. When undefined equals None
        self.__acc = None  # The final training accuracy. When undefined equals None

    # Define functions for accessing the private variables
    def center(self):
        return self.__center

    def matrix(self):
        return self.__matrix

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

    def find_matrix(self, X, Y, verbose=True):
        """
        Optimize for the characteristic matrix A - that is solve the constrained nonlinear optimization problem

            min. sum(u_i for i in 1, 2, ... , |X|) + sum(v_i for i in 1, 2, ... , |Y|) + gamma * ||w||^2
            s.t. u_i - (x_i - c)^T A(x_i - c) >= 0      for i in 1, 2, ... , |X|
                 v_i + (y_i - c)^T A(y_i - c) - 2 >= 0  for i in 1, 2, ... , |Y|
                 u_i >= 0                               for i in 1, 2, ... , |X|
                 v_i >= 0                               for i in 1, 2, ... , |Y|

            where u_i and v_i are variables created from the relaxation of the original condition,
            c is the center point of the ball, gamma is the robustness factor and w is a vector of form
            w = [lambda_1^(-1/2) lambda_2^(-1/2) ... lambda_n^(-1/2)]^T

        NOTE! For A to define an ellipse must it be a s.p.d matrix. In the case of this function A will be a diagonal
        matrix with all positive elements on the diagonal

        :param X: The data points with the correct label. A numpy.ndarray of shape (m, d)
        :param Y: The data points with the incorrect labels. A numpy.ndarray of shape (n-m, d)
        :param verbose: Boolean telling whether the results of the optimization are printed
        :return: Void
        """
        c = self.center()
        assert c is not None
        d = X.shape[1]
        m = X.shape[0]
        n = Y.shape[0]
        gamma = (n + m) * self.base_gamma

        # Define the objective function
        def obj(params):
            return sum(params[d:m+d]) + \
                   sum(params[m+d:m+n+d]) + \
                   gamma * sum(params[0:d] ** (-1))

        cons = []
        # Define the constraints for x_i in X
        for i in range(0, m):
            cons.append({'type': 'ineq',
                         'fun': lambda params, i=i: params[i + d] - np.dot(X[i] - c, params[0:d] * (X[i] - c))})
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: params[i + d]})

        # Define the constraints for y_i in Y
        for i in range(0, n):
            cons.append({'type': 'ineq',
                         'fun': lambda params, i=i: params[i + d + m] + np.dot(Y[i] - c, params[0:d] * (Y[i] - c)) - 2})
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: params[i + d + m]})

        # Define constraints for the diagonal of A
        for i in range(0, d):
            cons.append({'type': 'ineq', 'fun': lambda params, i=i: params[i]})

        # Optimize for r
        variables = np.ones((d + m + n,))
        res = minimize(obj, variables, method='SLSQP', constraints=tuple(cons))

        if verbose:
            print(f"Results of optimizing for label: '{self.label.rstrip()}'")
            if res.success:
                print(f"Success: {res.success}")
            else:
                print(f"Success: {res.success}")
                print(f"Reason: {res.message}")
            print(f"Found optimum: {res.fun}")
            print(f"Found diagonal: {res.x[0:d]}")
            print(f"Other variables:\n{res.x[d:]}")
            print()

        self.__matrix = np.diag(res.x[0:d])

        # Go through the points in X and Y and compute the accuracy
        self.__acc = (sum([1 for x in X if self.in_ellipsoid(x)]) + sum([1 for y in Y if not self.in_ellipsoid(y)])) / (m + n)

    def in_ellipsoid(self, point):
        """
        Checks if the inputted point falls within the ball defined in the node
        :param point: Input point. A numpy.ndarray of shape (d,)
        :return: Boolean
        """
        return np.matmul((point - self.center()).T, np.matmul(self.matrix(), (point - self.center()))) <= 1

    def dist(self, point):
        """
        Computes the distance from the surface of the ellipse to the given point
        :param point:
        :return:
        """
        return np.matmul((point - self.center()).T, np.matmul(self.matrix(), (point - self.center())))


