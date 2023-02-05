"""
Collection of functions for visualizing the results of the classifier
"""
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ellipse_classifier import EllipseClassifier
from ball_classifier import BallClassifier

colors = ['red', 'blue', 'green', 'grey', 'magenta', 'cyan', 'orange', 'deeppink', 'brown', 'darkviolet']


def viz_classic_neural_net(y, X, neural_net_shape, labels,
                           axis_names=("$X_1$", "$X_2$"),
                           verbose=True,
                           show=False,
                           save_path="neural_net.png",
                           y_test=None, X_test=None):
    """
    This function trains a classic neural network with given data and plots the resulting
    decision boundary together with the datapoints. Note, feature space must be 2 dimensional.
    Inspired by: https://github.com/mediumssharma23/avisualintroductiontonn/blob/master/visual_nn.ipynb
    :param y: Labels as numpy array with shape (n, 1) or (n, )
    :param X: Features as numpy array with shape (n, 2)
    :param neural_net_shape: The shape of wanted hidden layers as list of int
    :param labels: List of distinct labels
    :param axis_names: Tuple of type (x_label, y_label) (default: ($X_1$, $X_2$))
    :param verbose: Boolean that tells whether function should print progress (default: True)
    :param show: Boolean to choose whether image is shown or saved directly
    :param save_path: Path to where drawn figure should be saved as str (default: "neural_net.png")
    :param y_test: Labels for a test set of points. If not passed testing will not be done.
    :param X_test: Features for a test set of points. If not passed testing will not be done.
    :return: Void
    """
    assert X.shape[1] == 2
    assert X.shape[0] == y.shape[0]

    # Train the model
    mlp = MLPClassifier(hidden_layer_sizes=neural_net_shape,
                        learning_rate_init=0.005,
                        activation='logistic').fit(X, y)

    # Get the maxima and minima for the columns of X
    maxima = np.max(X, axis=0)
    minima = np.min(X, axis=0)

    # Create bounds for data to find the decision boundary
    num_points = 200
    xx = np.linspace(minima[0], maxima[0], num_points)
    yy = np.linspace(minima[1], maxima[1], num_points)

    # Use the trained model to predict labels for points defined above
    point_xs = []
    point_ys = []
    predictions = []
    count = 0
    for point_x in xx:
        for point_y in yy:
            point_xs.append(point_x)
            point_ys.append(point_y)
            predictions.append(mlp.predict(np.array([point_x, point_y]).reshape(1, -1)))

        if verbose:
            print(f"{(count / num_points) * 100}% of mapping done")
        count += 1

    # Generate color for each label
    label_to_color = {}
    if len(labels) <= 10:
        for i, label in enumerate(labels):
            label_to_color[label] = colors[i]

    else:
        raise ValueError("Number of distinct labels exceeds 10")

    prediction_colors = [label_to_color[label[0]] for label in predictions]

    # Plot everything
    if verbose:
        print("\nDrawing figure...")

    fig, ax = plt.subplots()

    ax.scatter(point_xs, point_ys, c=prediction_colors, s=50, alpha=0.01, lw=0)

    # Sort the data points by label
    tuples = [(y[i], X[i][0], X[i][1]) for i in range(y.shape[0])]
    for label in labels:
        data_points = np.array([[tup[1], tup[2]] for tup in tuples if tup[0] == label])
        ax.scatter(data_points[:, 0], data_points[:, 1], c=label_to_color[label], s=50,
                   edgecolor="white", linewidth=1, label=label)

    ax.set(xlim=(minima[0] - 1, maxima[0] + 1), ylim=(minima[1] - 1, maxima[1] + 1),
           xlabel=axis_names[0], ylabel=axis_names[1])

    ax.legend(loc='lower right')

    if show:
        plt.show()
    else:
        fig.savefig(save_path)

    if X_test is not None and y_test is not None:
        assert X_test.shape[1] == 2
        assert X_test.shape[0] == y_test.shape[0]

        # Predict the points in testing dataset
        y_hat = mlp.predict(X_test)

        # Count correct classifications
        correct_classifications = [1 for y_actual, y_pred in zip(y_test, y_hat) if y_actual == y_pred]

        print("\nMLP prediction accuracy is:", len(correct_classifications) / y_test.shape[0], '\n')


def viz_knearest_neighbors(y, X, labels, n_neighbors,
                           weights="uniform",
                           axis_names=("$X_1$", "$X_2$"),
                           verbose=True,
                           show=False,
                           save_path="kneighbors.png",
                           y_test=None, X_test=None):
    # Train the model
    knn = KNeighborsClassifier(n_neighbors, weights=weights).fit(X, y)

    # Get the maxima and minima for the columns of X
    maxima = np.max(X, axis=0)
    minima = np.min(X, axis=0)

    # Create bounds for data to find the decision boundary
    num_points = 200
    xx = np.linspace(minima[0], maxima[0], num_points)
    yy = np.linspace(minima[1], maxima[1], num_points)

    # Use the trained model to predict labels for points defined above
    point_xs = []
    point_ys = []
    predictions = []
    count = 0
    for point_x in xx:
        for point_y in yy:
            point_xs.append(point_x)
            point_ys.append(point_y)
            predictions.append(knn.predict(np.array([point_x, point_y]).reshape(1, -1)))

        if verbose:
            print(f"{(count / num_points) * 100}% of mapping done")
        count += 1

    # Generate color for each label
    label_to_color = {}
    if len(labels) <= 10:
        for i, label in enumerate(labels):
            label_to_color[label] = colors[i]

    else:
        raise ValueError("Number of distinct labels exceeds 10")

    prediction_colors = [label_to_color[label[0]] for label in predictions]

    # Plot everything
    if verbose:
        print("\nDrawing figure...")

    fig, ax = plt.subplots()

    ax.scatter(point_xs, point_ys, c=prediction_colors, s=50, alpha=0.01, lw=0)

    # Sort the data points by label
    tuples = [(y[i], X[i][0], X[i][1]) for i in range(y.shape[0])]
    for label in labels:
        data_points = np.array([[tup[1], tup[2]] for tup in tuples if tup[0] == label])
        ax.scatter(data_points[:, 0], data_points[:, 1], c=label_to_color[label], s=50,
                   edgecolor="white", linewidth=1, label=label)

    ax.set(xlim=(minima[0] - 1, maxima[0] + 1), ylim=(minima[1] - 1, maxima[1] + 1),
           xlabel=axis_names[0], ylabel=axis_names[1])

    ax.legend(loc='lower right')

    if show:
        plt.show()
    else:
        fig.savefig(save_path)

    if X_test is not None and y_test is not None:
        assert X_test.shape[1] == 2
        assert X_test.shape[0] == y_test.shape[0]

        # Predict the points in testing dataset
        y_hat = knn.predict(X_test)

        # Count correct classifications
        correct_classifications = [1 for y_actual, y_pred in zip(y_test, y_hat) if y_actual == y_pred]

        print("\nKNN prediction accuracy is:", len(correct_classifications) / y_test.shape[0], '\n')


def viz_ball_classifier(classifier, X, y,
                        axis_names=("x", "y"),
                        show=False,
                        save_path="ball_classifier.png",
                        title_with_gamma=True):
    """
    Plots the balls (circles) representing each node and the  datapoints passed as
    argument to visualize the performance of the classifier
    :param classifier:
    :param X:
    :param y:
    :param axis_names:
    :param save_path:
    :param title_with_gamma:
    :return:
    """
    assert isinstance(classifier, BallClassifier)

    # Find the distinct labels
    labels = list(classifier.nodes.keys())

    # Get the maxima and minima for the columns of X
    maxima = np.max(X, axis=0)
    minima = np.min(X, axis=0)

    max_limit = max(maxima)
    min_limit = min(minima)

    # Generate color for each label
    label_to_color = {}
    if len(labels) <= 10:
        for i, label in enumerate(labels):
            label_to_color[label] = colors[i]

    else:
        raise ValueError("Number of distinct labels exceeds 10")

    fig, ax = plt.subplots()

    # Plot the circles representing the region allocated for each node
    for node in classifier.nodes.values():
        label_color = label_to_color[node.label]
        circle = plt.Circle(node.center(), radius=node.radius(),
                            color=label_color,
                            alpha=0.3, linewidth=1)
        ax.add_patch(circle)

    # Sort the data points by label
    tuples = [(y[i], X[i][0], X[i][1]) for i in range(y.shape[0])]
    for label in labels:
        data_points = np.array([[tup[1], tup[2]] for tup in tuples if tup[0] == label])
        ax.scatter(data_points[:, 0], data_points[:, 1], c=label_to_color[label], s=50,
                   edgecolor="white", linewidth=1, label=label)

    ax.set(xlim=(min_limit - 1, max_limit + 1), ylim=(min_limit - 1, max_limit + 1),
           xlabel=axis_names[0], ylabel=axis_names[1])

    ax.legend(loc='lower right')

    if title_with_gamma:
        ax.set_title(f"Used robustness factor: $\gamma$ = {classifier.base_gamma}")
        if show:
            plt.show()
        else:
            fig.savefig(save_path.split('.')[0] + f"_gamma_{classifier.base_gamma}.png")

    else:
        if show:
            plt.show()
        else:
            fig.savefig(save_path)


def viz_ellipse_classifier(classifier, X, y,
                           axis_names=("x", "y"),
                           show=False,
                           save_path="ellipse_classifier.png",
                           title_with_gamma=True):
    """
    TODO: DESCRIPTION
    :param classifier:
    :param X:
    :param y:
    :param axis_names:
    :param save_path:
    :param title_with_gamma:
    :return:
    """
    assert isinstance(classifier, EllipseClassifier)

    # Find the distinct labels
    labels = list(classifier.nodes.keys())

    # Get the maxima and minima for the columns of X
    maxima = np.max(X, axis=0)
    minima = np.min(X, axis=0)

    max_limit = max(maxima)
    min_limit = min(minima)

    # Generate color for each label
    label_to_color = {}
    if len(labels) <= 10:
        for i, label in enumerate(labels):
            label_to_color[label] = colors[i]

    else:
        raise ValueError("Number of distinct labels exceeds 10")

    fig, ax = plt.subplots()

    # Plot the circles representing the region allocated for each node
    for node in classifier.nodes.values():
        label_color = label_to_color[node.label]
        width = 2 * node.matrix()[0, 0] ** (-1/2)
        height = 2 * node.matrix()[1, 1] ** (-1/2)
        # width = node.matrix()[0, 0]
        # height = node.matrix()[1, 1]
        ellipse = Ellipse(node.center(), width=width, height=height,
                          color=label_color, alpha=0.3, linewidth=1)
        ax.add_patch(ellipse)

    # Sort the data points by label
    tuples = [(y[i], X[i][0], X[i][1]) for i in range(y.shape[0])]
    for label in labels:
        data_points = np.array([[tup[1], tup[2]] for tup in tuples if tup[0] == label])
        ax.scatter(data_points[:, 0], data_points[:, 1], c=label_to_color[label], s=50,
                   edgecolor="white", linewidth=1, label=label)

    ax.set(xlim=(min_limit - 1, max_limit + 1), ylim=(min_limit - 1, max_limit + 1),
           xlabel=axis_names[0], ylabel=axis_names[1])

    ax.legend(loc='lower right')

    if title_with_gamma:
        ax.set_title(f"Used robustness factor: $\gamma$ = {classifier.base_gamma}")
        if show:
            plt.show()
        else:
            fig.savefig(save_path.split('.')[0] + f"_gamma_{classifier.base_gamma}.png")

    else:
        if show:
            plt.show()
        else:
            fig.savefig(save_path)
