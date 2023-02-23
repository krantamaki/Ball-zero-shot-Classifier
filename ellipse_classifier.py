"""
Class defining the whole ellipse classifier and associated functions
"""
import numpy as np
from numpy.linalg import norm
from scipy.special import softmax
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Process, Manager
from ellipse_node import EllipseNode


class EllipseClassifier:

    def __init__(self, empty_space_label="Empty space", base_gamma=1, y_multiple=5, scaling_factor=1):
        # The label used to define empty space if prediction happened to be it
        self.empty_space_label = empty_space_label

        # Parameter for balancing the need for robustness and correctness
        self.base_gamma = base_gamma

        # As there might be a lot of distinct labels the set Y can go very large so define a multiple that
        # signifies how many times more points should be in Y compared to X
        self.y_multiple = y_multiple

        # As the data points might be very close by (as is the case with MEG data) it might be worthwhile to
        # rescale the points to make the optimization easier for the solvers. Rescaling is done as c * I * x
        # for every datapoint x, where c is the scaling factor
        self.scaling_factor = scaling_factor

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

    def __train_node__(self, label, grouped_points):
        """
        Helper function that wraps multiple operations within it
        :param label: The label of the node to be trained
        :param grouped_points: The datapoints grouped by label
        str -> points: numpy.ndarray of shape (m, d) containing all points and labels
        :return: Void
        """
        X = grouped_points[label]

        # Combine the rest of the points into one numpy array
        Y = np.concatenate([value for key, value in grouped_points.items() if key != label])

        # Take a random sample from set Y
        if Y.shape[0] > X.shape[0] * self.y_multiple:
            Y = Y[np.random.randint(Y.shape[0], size=X.shape[0] * self.y_multiple), :]

        # Rescale the datapoints
        scaler = self.scaling_factor * np.identity(X.shape[1])
        X = np.array([np.matmul(scaler, x) for x in X])
        Y = np.array([np.matmul(scaler, y) for y in Y])

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
        self.data_dim = data.shape[1]

        # Group datapoints by label
        grouped_points = self.__group_points__(data, labels)

        # Create a node for each label and train with the datapoints
        for label in grouped_points:
            self.__train_node__(label, grouped_points)

    def __par_train_node__(self, label, grouped_points, ret_dict):
        """
        Helper function that wraps multiple operations within it so that they can be called in parallel pool
        :param label: The label of the node to be trained
        :param grouped_points: The datapoints grouped by label
        :param ret_dict: multiprocessing.Manager.dict() to store the return value
        :return: Void
        """
        X = grouped_points[label]

        # Combine the rest of the points into one numpy array
        Y = np.concatenate([value for key, value in grouped_points.items() if key != label])

        # Take a random sample from set Y
        if Y.shape[0] > X.shape[0] * self.y_multiple:
            Y = Y[np.random.randint(Y.shape[0], size=X.shape[0] * self.y_multiple), :]

        # Rescale the datapoints
        scaler = self.scaling_factor * np.identity(X.shape[1])
        X = np.array([np.matmul(scaler, x) for x in X])
        Y = np.array([np.matmul(scaler, y) for y in Y])

        # Train the node
        new_node = EllipseNode(label, base_gamma=self.base_gamma)
        new_node.find_center(X)
        new_node.find_matrix(X, Y)
        ret_dict[label] = new_node

    def par_train(self, data, labels):
        """
        Train the model by optimizing the balls for each individual node in parallel.
        :param data: The data array. A numpy.ndarray of shape (n, d)
        :param labels: The label array. A numpy.ndarray of shape (n,)
        :return: Void
        """
        self.data_dim = data.shape[1]

        # Group datapoints by label
        grouped_points = self.__group_points__(data, labels)

        # Initialize processes
        manager = Manager()
        ret_dict = manager.dict()
        processes = []
        for label in grouped_points:
            process = Process(target=self.__par_train_node__, args=(label, grouped_points, ret_dict))
            processes.append(process)
            process.start()

        # Complete the processes
        for proc in processes:
            proc.join()

        self.nodes = ret_dict

    def eval(self):
        """
        Evaluate the total training accuracy of the model
        :return: The total training accuracy as a floating point number
        """
        assert len(self.nodes) > 0
        return sum([node.accuracy() for label, node in self.nodes.items()]) / len(self.nodes)

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
            for label, node in self.nodes.items():
                if label not in y:
                    raise RuntimeError("\nSemantic vector not provided for all existing labels")

        # Use the given arrays to create a dictionary and store it in memory
        sem_dict = {}
        for i in range(0, y.shape[0]):
            sem_dict[y[i]] = S[i]

        self.semantic_vectors = sem_dict

    def add_diagonals_and_centers(self, diags, centers, labels):
        """
        Function for generating the nodes for already existing diagonals and centerpoints
        :param diags:
        :param centers:
        :param labels:
        :return: Void
        """
        assert diags.shape == centers.shape
        for i in range(len(labels)):
            new_node = EllipseNode(labels[i])
            new_node.add_diagonal_and_center(diags[i], centers[i])
            self.nodes[labels[i]] = new_node

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
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='cosine').fit(S)
        d, i = nbrs.kneighbors(x)

        # Return the label corresponding with the
        return y[i[0]]

    def sem_test(self, X_test, y_test):
        assert len(self.nodes) != 0
        assert len(self.semantic_vectors) != 0
        assert X_test.shape[0] == y_test.shape[0]
        correct = 0

        # Scale the datapoints in X_test
        if self.scaling_factor != 1:
            scaler = self.scaling_factor * np.identity(X_test.shape[1])
            X_test = np.array([np.matmul(scaler, x) for x in X_test])

        # Go over the testing points
        for i in range(y_test.shape[0]):
            point = X_test[i]
            correct_label = y_test[i]

            # Compute the distances from the point to the surface of each of the balls
            dists = [(label, node.dist(point)) for label, node in self.nodes.items()]

            labels = [tup[0] for tup in dists]
            dists = [tup[1] for tup in dists]

            # Convert the distances to weights
            weights = [max(dists) - dist for dist in dists]
            weights = softmax(weights)
            # weights = [(dist + 1) ** (-1) for dist in dists]
            # weights = softmax(weights)

            # Compute the weighted average of the semantic vectors
            avg = np.zeros((self.sem_dim,))

            for i, label in enumerate(labels):
                avg += weights[i] * self.semantic_vectors[label]

            # Do a brute force search for the 1-nearest neighbour
            """
            best_dist = -float('inf')
            best_label = ""
            for label, vect in self.semantic_vectors.items():
                if 1 - cosine(avg, vect) > best_dist:
                    best_label = label
                    best_dist = 1 - cosine(avg, vect)

            if best_label.strip() == correct_label.strip():
                if correct_label not in self.nodes:
                    print("CORRECT ZERO-SHOT PREDICTION\n")
                correct += 1
            else:
                if correct_label not in self.nodes:
                    print("INCORRECT ZERO-SHOT PREDICTION")
                print(f"Point to predict: {point}")
                print(f"Predicted label: {best_label.strip()}; Correct label: {correct_label.strip()}")
                print(f"Distance to predicted label: {1 - cosine(avg, self.semantic_vectors[best_label])}")
                print(f"Distance to correct label: {1 - cosine(avg, self.semantic_vectors[correct_label])}\n")
            """
            best_dist = float('inf')
            best_label = ""
            for label, vect in self.semantic_vectors.items():
                if norm(avg - vect) < best_dist:
                    best_label = label
                    best_dist = norm(avg - vect)

            if best_label == correct_label:
                if correct_label not in self.nodes:
                    print("CORRECT ZERO-SHOT PREDICTION\n")
                correct += 1
            else:
                if correct_label not in self.nodes:
                    print("INCORRECT ZERO-SHOT PREDICTION")
                # print(f"Point to predict: {point}")
                print(f"Predicted label: {best_label}; Correct label: {correct_label}")
                # print(f"Distance to predicted label: {norm(avg - self.semantic_vectors[best_label])}")
                # print(f"Distance to correct label: {norm(avg - self.semantic_vectors[correct_label])}\n")

        # Return the proportion of correct predictions
        if y_test.shape[0] != 0:
            return correct / y_test.shape[0]

        return 0.0

    def test(self, X_test, y_test):
        assert len(self.nodes) != 0
        assert X_test.shape[0] == y_test.shape[0]
        correct = 0

        # Scale the datapoints in X_test
        if self.scaling_factor != 1:
            scaler = self.scaling_factor * np.identity(X_test.shape[1])
            X_test = np.array([np.matmul(scaler, x) for x in X_test])

        # Go over the testing points
        for i in range(y_test.shape[0]):
            point = X_test[i]
            correct_label = y_test[i]

            # Compute the distances from the point to the surface of each of the balls
            dists = [(label, node.dist(point)) for label, node in self.nodes.items()]

            labels = [tup[0] for tup in dists]
            dists = [tup[1] for tup in dists]

            # Convert the distances to weights
            weights = [max(dists) - dist for dist in dists]
            weights = softmax(weights)
            # weights = [(dist + 1) ** (-1) for dist in dists]
            # weights = softmax(weights)

            # Compute the weighted average of the semantic vectors
            avg = np.zeros((X_test.shape[1],))

            for i, label in enumerate(labels):
                avg += weights[i] * self.nodes[label].center()

            # Do a brute force search for the 1-nearest neighbour
            """
            best_dist = -float('inf')
            best_label = ""
            for label, vect in self.semantic_vectors.items():
                if 1 - cosine(avg, vect) > best_dist:
                    best_label = label
                    best_dist = 1 - cosine(avg, vect)

            if best_label.strip() == correct_label.strip():
                if correct_label not in self.nodes:
                    print("CORRECT ZERO-SHOT PREDICTION\n")
                correct += 1
            else:
                if correct_label not in self.nodes:
                    print("INCORRECT ZERO-SHOT PREDICTION")
                print(f"Point to predict: {point}")
                print(f"Predicted label: {best_label.strip()}; Correct label: {correct_label.strip()}")
                print(f"Distance to predicted label: {1 - cosine(avg, self.semantic_vectors[best_label])}")
                print(f"Distance to correct label: {1 - cosine(avg, self.semantic_vectors[correct_label])}\n")
            """
            best_dist = float('inf')
            best_label = ""
            for label in self.nodes:
                vect = self.nodes[label].center()
                if norm(avg - vect) < best_dist:
                    best_label = label
                    best_dist = norm(avg - vect)

            if best_label == correct_label:
                if correct_label not in self.nodes:
                    print("CORRECT ZERO-SHOT PREDICTION\n")
                correct += 1
            else:
                if correct_label not in self.nodes:
                    print("INCORRECT ZERO-SHOT PREDICTION")
                # print(f"Point to predict: {point}")
                print(f"Predicted label: {best_label}; Correct label: {correct_label}")
                # print(f"Distance to predicted label: {norm(avg - self.nodes[best_label].center())}")
                # print(f"Distance to correct label: {norm(avg - self.nodes[correct_label].center())}\n")

        # Return the proportion of correct predictions
        return correct / y_test.shape[0]

