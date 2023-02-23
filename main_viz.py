import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from multiprocessing import freeze_support
from timeit import default_timer as timer
from datetime import timedelta
from sklearn.neighbors import KNeighborsClassifier
# from ellipse_classifier import EllipseClassifier
from ellipse_node import EllipseNode
from visualize import viz_ellipse_classifier, viz_classic_neural_net, viz_knearest_neighbors
from ball_classifier import BallClassifier
from leave_n_out_split import leave_n_out_split


def gen_points(characteristic_matrix, mean, deviation, n):
    X = []
    Y = []

    # Generate points until both X and Y have n points in them
    while len(X) < n or len(Y) < n:
        x1 = np.random.normal(mean[0], deviation[0], size=2 * n)
        x2 = np.random.normal(mean[1], deviation[1], size=2 * n)
        points = np.column_stack((x1, x2))

        dists = np.array([np.dot(points[i] - mean, np.matmul(characteristic_matrix, points[i] - mean)) for i in range(2 * n)])

        X_add = points[np.where(dists < 1)]
        Y_add = points[np.where((dists > 1) & (dists < 2))]

        X += X_add.tolist()
        Y += Y_add.tolist()

    return np.array(X[0:n]), np.array(Y[0:n])


def main():
    mat = np.array([[1.2, 0.], [0., 3]])
    mean = np.array([0., 0.])
    deviation = np.array([1.5, 4])
    num_points = 100

    X, Y = gen_points(mat, mean, deviation, num_points)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X[:, 0], X[:, 1], label="X")
    ax.scatter(Y[:, 0], Y[:, 1], label="Y")

    ax.legend()
    plt.show()

    node = EllipseNode(label="X", base_gamma=0)
    node.find_center(X)
    node.find_matrix(X, Y)

    mod_node = EllipseNode(label="X", base_gamma=1)
    mod_node.find_center(X)
    mod_node.find_matrix(X, Y, mod=True)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    width = 2 * node.matrix()[0, 0] ** (-1 / 2)
    height = 2 * node.matrix()[1, 1] ** (-1 / 2)
    ellipse = Ellipse(node.center(), width=width, height=height, alpha=0.3, linewidth=1)
    ax1.add_patch(ellipse)

    ax1.scatter(X[:, 0], X[:, 1], label="$X$")
    ax1.scatter(Y[:, 0], Y[:, 1], label="$Y$")

    mod_width = 2 * mod_node.matrix()[0, 0] ** (-1 / 2)
    mod_height = 2 * mod_node.matrix()[1, 1] ** (-1 / 2)
    mod_ellipse = Ellipse(mod_node.center(), width=mod_width, height=mod_height, alpha=0.3, linewidth=1)
    ax2.add_patch(mod_ellipse)

    ax2.scatter(X[:, 0], X[:, 1], label="$X$")
    ax2.scatter(Y[:, 0], Y[:, 1], label="$Y$")

    ax1.legend()
    ax1.title.set_text("$\gamma_{0} = 0$")
    ax2.legend()
    ax2.title.set_text("$\gamma_{0} = 1$")
    plt.show()


if __name__ == '__main__':
    freeze_support()
    main()



