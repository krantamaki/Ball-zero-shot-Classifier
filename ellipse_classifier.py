"""
Class defining the whole ellipse classifier and associated functions
"""
import numpy as np
from multiprocessing import Pool
from ellipse_node import EllipseNode


# TODO: ERROR HANDLING

class EllipseClassifier:

    def __init__(self, empty_space_label="Empty space", base_gamma=1):
        # The label used to define empty space if prediction happened to be it
        self.empty_space_label = empty_space_label

        # Parameter for balancing the need for robustness and correctness
        self.base_gamma = base_gamma

        # Dictionary of nodes of form label: str -> node: Node
        self.nodes = {}

    def predict(self, point):
        """
        Find the node into which the inputted point falls (if such exists). If multiple is found
        select the one which has the center closest to the inputted point. If no node is found
        return the label depicting empty space
        :param point: Input point. A numpy.ndarray of shape (d,)
        :return: Label of the prediction
        """
        ret_label = ""
        shortest_dist_found = float("inf")
        for label, node in self.nodes.items():
            found_dist = node.in_ball_with_dist(point)
            if 0.0 < found_dist < shortest_dist_found:
                ret_label = label
                shortest_dist_found = found_dist

        return ret_label if ret_label != "" else self.empty_space_label

    def __group_points__(self, X, y):
        """
        Helper function used by the training functions that groups the datapoints by their
        labels into a dictionary of form label -> datapoints
        :param X: The data array. A numpy.ndarray of shape (n, d)
        :param y: The label array. A numpy.ndarray of shape (n,)
        :return: Dictionary of form label: str -> points: numpy.ndarray of shape (m, d)
        """
        ret_dict = {}
        # Get the distinct labels
        unique_labels = np.unique(y)

        # Loop over the labels
        for label in unique_labels:
            # Find the indexes at which the label in question is found in array y
            label_indexes = (y == label).nonzero()[0]

            # Get the corresponding datapoints and update the dictionary
            ret_dict[label] = X[label_indexes]

        return ret_dict

    def __train_node__(self, params):
        """
        Helper function that wraps multiple operations within it so that they can be called in parallel pool
        :param params: Tuple holding the label of the node in question and a dictionary of form label:
        str -> points: numpy.ndarray of shape (m, d) containing all points and labels
        :return: Void
        """
        label = params[0]
        grouped_points = params[1]

        # Check that the model hasn't yet been trained with the label
        # assert label not in [node.label for node in self.nodes]

        X = grouped_points[label]

        # Combine the rest of the points into one numpy array
        Y = np.concatenate([value for key, value in grouped_points.items() if key != label])

        # Train the node
        new_node = EllipseNode(label, base_gamma=self.base_gamma)
        new_node.find_center(X)
        new_node.find_matrix(X, Y)
        self.nodes[label] = new_node

    def train(self, data, labels):
        """
        Train the model by sequentially optimizing the balls for each individual node.
        :param data: The data array. A numpy.ndarray of shape (n, d)
        :param labels: The label array. A numpy.ndarray of shape (n,)
        :return: Void
        """
        # Group datapoints by label
        grouped_points = self.__group_points__(data, labels)

        # Create a node for each label and train with the datapoints
        for label, X in grouped_points.items():
            params = [label, grouped_points]
            self.__train_node__(params)

    def par_train(self, data, labels):
        """
        TODO: FIX BUG
        Train the model by optimizing the balls for each individual node in parallel.
        :param data: The data array. A numpy.ndarray of shape (n, d)
        :param labels: The label array. A numpy.ndarray of shape (n,)
        :return: Void
        """
        # Group datapoints by label
        grouped_points = self.__group_points__(data, labels)
        params = [[label, grouped_points] for label in grouped_points]

        # Train the nodes in parallel
        with Pool() as pool:
            pool.imap_unordered(self.__train_node__, params)

