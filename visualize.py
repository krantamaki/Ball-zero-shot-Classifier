"""
Collection of functions for visualizing the results of the classifier
"""
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

colors = ['red', 'blue', 'green', 'grey', 'magenta', 'cyan', 'orange', 'deeppink', 'brown', 'darkviolet']


def viz_classic_neural_net(y, X, neural_net_shape, labels,
                           axis_names=("$X_1$", "$X_2$"),
                           verbose=True,
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

    fig.savefig(save_path)

    if X_test is not None and y_test is not None:
        assert X_test.shape[1] == 2
        assert X_test.shape[0] == y_test.shape[0]

        # Predict the points in testing dataset
        y_hat = mlp.predict(X_test)

        # Count correct classifications
        correct_classifications = [1 for y_actual, y_pred in zip(y_test, y_hat) if y_actual == y_pred]

        print("\nMLP prediction accuracy is:", len(correct_classifications) / y_test.shape[0])


def viz_classifier(classifier, X, y):
    """

    :param classifier:
    :param X:
    :param y:
    :return:
    """
