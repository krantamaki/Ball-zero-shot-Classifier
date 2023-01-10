"""
Class defining the whole ellipse classifier and associated functions
"""
import numpy as np
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
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

        # Dictionary of form label: str -> vector: np.ndarray for holding the semantic space information
        self.semantic_vectors = {}

        # Helpful constants
        self.data_dim = None
        self.sem_dim = None

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
        self.data_dim = data.shape[1]

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

    def add_sematic_vectors(self, S, y):
        """
        Adds the semantic vectors into memory for use in zero-shot learning and predicting
        Note! Each label used in training should in the label array and each label can be exactly
        once in the label array
        :param S: The semantic data array. A numpy.ndarray of shape (n0, s)
        :param y: The label array. A numpy.ndarray of shape (n0,)
        :return: Void
        """
        assert y.shape[0] == np.unique(y).shape[0]
        assert S.shape[0] == y.shape[0]
        self.sem_dim = S.shape[1]

        # Check that there is a semantic vector for each of the possibly existing nodes
        if len(self.nodes) != 0:
            for node in self.nodes:
                if node.label not in y:
                    raise RuntimeError("\nSemantic vector not provided for all existing labels")

        # Use the given arrays to create a dictionary and store it in memory
        sem_dict = {}
        for i in range(0, y.shape[0]):
            sem_dict[y[i]] = S[i]

        self.semantic_vectors = sem_dict

    def sem_predict(self, point):
        """
        UNTESTED
        Find the node into which the inputted point falls. Firstly find the approximate semantic vectors for the point
        and then se 1-nearest neighbour search to find the label
        :param point: Input point. A numpy.ndarray of shape (d,)
        :return: Label of the prediction
        """
        assert len(self.nodes) != 0
        assert len(self.semantic_vectors) != 0

        # Compute the distances from the point to the surface of each of the balls
        dists = [(node.label, node.dist(point)) for node in self.nodes]

        labels = [tup[0] for tup in dists]
        dists = [tup[1] for tup in dists]

        # Convert the distances to weights
        """
        weights = [max(dists) - dist for dist in dists]
        weights = softmax(weights)
        """
        weights = [(dist + 1) ** (-1) for dist in dists]
        weights = softmax(weights)

        # Compute the weighted average of the semantic vectors
        avg = np.zeros((self.sem_dim,))
        for i, label in enumerate(labels):
            avg += weights[i] * self.semantic_vectors[label]

        # Reconstruct the semantic space matrix and the label vector
        S = []
        y = []
        for label, vector in self.semantic_vectors.items():
            S.append(vector)
            y.append(label)

        S = np.array(S)
        y = np.array(y)

        # Wrap the average in a numpy array
        x = np.array([avg])

        # Find the 1-nearest neighbour
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(S)
        d, i = nbrs.kneighbors(x)

        # Return the label corresponding with the
        return y[i[0]]

    def sem_test(self, X_test, y_test):
        """
        UNTESTED
        Compute the testing accuracy for a given testing dataset
        :param X_test: The testing data array. A numpy.ndarray of shape (n, d)
        :param y_test: The testing label array. A numpy.ndarray of shape (n,)
        :return: The testing accuracy as a float
        """
        assert len(self.nodes) != 0
        assert len(self.semantic_vectors) != 0
        assert X_test.shape[0] == y_test.shape[0]

        # Reconstruct the semantic space matrix and the label vector and train the 1-nearest neighbour mode
        S = []
        y = []
        for label, vector in self.semantic_vectors.items():
            S.append(vector)
            y.append(label)

        S = np.array(S)
        y = np.array(y)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(S)

        # Initialize a counter for correct predictions
        correct = 0

        # Go over the testing points
        for i in range(y.shape[0]):
            point = X_test[i]
            correct_label = y_test[i]

            # Compute the distances from the point to the surface of each of the balls
            dists = [(node.label, node.dist(point)) for node in self.nodes]

            labels = [tup[0] for tup in dists]
            dists = [tup[1] for tup in dists]

            # Convert the distances to weights
            """
            weights = [max(dists) - dist for dist in dists]
            weights = softmax(weights)
            """
            weights = [(dist + 1) ** (-1) for dist in dists]
            weights = softmax(weights)

            # Compute the weighted average of the semantic vectors
            avg = np.zeros((self.sem_dim,))

            for i, label in enumerate(labels):
                avg += weights[i] * self.semantic_vectors[label]

            # Wrap the average in a numpy array
            x = np.array([avg])

            # Find the 1-nearest neighbour
            d, i = nbrs.kneighbors(x)
            predicted_label = y[i[0]]

            if predicted_label == correct_label:
                correct += 1

        # Return the proportion of correct predictions
        return correct / y.shape[0]


