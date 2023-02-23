import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim
import spacy
from multiprocessing import freeze_support
from timeit import default_timer as timer
from datetime import timedelta
from sklearn.neighbors import KNeighborsClassifier
from ellipse_classifier import EllipseClassifier
from visualize import viz_ellipse_classifier, viz_classic_neural_net, viz_knearest_neighbors
from ball_classifier import BallClassifier
from leave_n_out_split import leave_n_out_split
from reduce_dimension_PCA import reduce_dimension_PCA
from reduce_dimension_TSNE import  reduce_dimension_TSNE


label_dict = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight",
              9: "Nine", 10: "T-shirt", 11: "Trousers", 12: "Pullover", 13: "Dress", 14: "Coat", 15: "Sandal",
              16: "Shirt", 17: "Sneaker", 18: "Bag", 19: "Boot"}

colors = ["gray", "lightcoral", "firebrick", "red", "coral", "sienna", "peru", "darkorange", "goldenrod", "darkkhaki",
          "olive", "yellow", "green", "lime", "turquoise", "teal", "steelblue", "navy", "blue", "blueviolet", "purple"]


def get_sample(df, label_col, label, n):
    if n == 'all':
        return df[df[label_col] == label].sample(frac=1, replace=False)

    return df[df[label_col] == label].sample(n, replace=False)


def sample_to_numpy(sample, excluded_cols):
    return sample.loc[:, ~sample.columns.isin(excluded_cols)].to_numpy()


def rescale(X, scaling_factor):
    scaler = scaling_factor * np.identity(X.shape[1])
    return np.array([np.matmul(scaler, x) for x in X])


def gen_semantic_vectors(df, dim, method='PCA'):
    vect_dict = {}

    # Go over the labels and generate a semantic vector for each based on all the images of the given label in the
    # training dataset
    for label in label_dict:
        X = sample_to_numpy(get_sample(df, "label", label, 'all'), ['label'])

        if method == 'PCA':
            sem_vect = np.mean(reduce_dimension_PCA(X, dim), axis=0)
        elif method == 'TSNE':
            sem_vect = np.mean(reduce_dimension_TSNE(X, dim), axis=0)
        else:
            raise RuntimeError("Improper method given")

        vect_dict[label] = sem_vect

    return vect_dict


def main1():
    # Load the data into dataframes
    df_mnist = pd.read_csv("Data/mnist_train.csv", sep=',')
    df_fashion = pd.read_csv("Data/fashion-mnist_train.csv", sep=',')

    # Add 10 to each label in the df_fashion dataframe so that the labels match those in label_dict
    df_fashion["label"] = df_fashion["label"] + 10

    # Change the column names of mnist set to match those of fashion mnist set
    col_dict = dict(zip(df_mnist.columns, df_fashion.columns))
    df_mnist.rename(columns=col_dict, inplace=True)

    # Concatenate the dataframes
    df = pd.concat([df_mnist, df_fashion], ignore_index=True)

    # Get num_points datapoints for each label
    num_points = 100

    X = []
    y = []

    for label in label_dict:
        X.append(sample_to_numpy(get_sample(df, 'label', label, num_points), ['label']))
        y.append([label] * num_points)

    X = np.concatenate(X)

    # print(X.shape)

    y = np.concatenate(y)

    # Plot the reduced dimension case for the points
    pca_X = reduce_dimension_PCA(X, 2)

    pca_X = rescale(pca_X, 10 ** (-3))

    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(label_dict)):
        ax.scatter(pca_X[:, 0][i * num_points:i * num_points + num_points // 2],
                   pca_X[:, 1][i * num_points:i * num_points + num_points // 2],
                   color=colors[i],
                   s=20,
                   label=label_dict[i])

    ax.legend()
    plt.show()

    # Do the test train split
    X_train, y_train, X_test, y_test = leave_n_out_split(X, y, 0, split_ratio=0.5)

    # Save the testing and training points as CSV
    df_train = pd.DataFrame(X_train)
    df_test = pd.DataFrame(X_test)

    df_train["label"] = y_train
    df_test["label"] = y_test

    df_train.to_csv("Data/comb-mnist_sample_train.csv", index=False)
    df_test.to_csv("Data/comb-mnist_sample_test.csv", index=False)


def main2():
    # Load the data into dataframes
    df_mnist = pd.read_csv("Data/mnist_train.csv", sep=',')
    df_fashion = pd.read_csv("Data/fashion-mnist_train.csv", sep=',')

    # Add 10 to each label in the df_fashion dataframe so that the labels match those in label_dict
    df_fashion["label"] = df_fashion["label"] + 10

    # Change the column names of mnist set to match those of fashion mnist set
    col_dict = dict(zip(df_mnist.columns, df_fashion.columns))
    df_mnist.rename(columns=col_dict, inplace=True)

    # Concatenate the dataframes
    df = pd.concat([df_mnist, df_fashion], ignore_index=True)

    vect_dict = gen_semantic_vectors(df, 100, method='PCA')

    print(vect_dict)

    # Get the data
    df_train = pd.read_csv("Data/comb-mnist_sample_train.csv")

    df_test = pd.read_csv("Data/comb-mnist_sample_test.csv")
    y_test = df_test["label"].to_numpy()
    X_test = df_test.loc[:, ~df_test.columns.isin(["label"])].to_numpy()

    sem_acc = []
    cen_acc = []
    knn_acc = []
    # left_out_acc = []

    # Go from leave 0 out until leave 4 out
    for i in range(0, 4 + 1):
        # Read the files with centers and diagonals
        centers = pd.read_csv(f"Data/comb_centers_{i}_left_out.csv")
        diags = pd.read_csv(f"Data/comb_diags_{i}_left_out.csv")

        known_labels = diags['label'].to_numpy()
        # left_out_labels = list(set(label_dict.keys()).difference(set(known_labels)))
        # print(left_out_labels)

        centers = centers.loc[:, ~centers.columns.isin(['label'])].to_numpy()
        diags = diags.loc[:, ~diags.columns.isin(['label'])].to_numpy()

        # Pass to model
        model = EllipseClassifier()
        model.add_diagonals_and_centers(diags, centers, known_labels)

        # Add semantic information to the model
        model.add_sematic_vectors(rescale(np.array(list(vect_dict.values())), 10 ** 15),
                                  np.array(list(vect_dict.keys())))

        # Test the model with all testing points using semantic vectors
        test_acc = model.sem_test(X_test, y_test)
        sem_acc.append(test_acc)

        # Test the model with all testing points using centers of the ellipsoids
        test_acc = model.test(X_test, y_test)
        cen_acc.append(test_acc)

        """
        # Test the point only with points from left out labels
        if i > 0:
            df_test_ = df_test.loc[df_test["label"].isin(left_out_labels), :]
            y_test_left_out = df_test_["label"].to_numpy()
            X_test_left_out = df_test_.loc[:, ~df_test_.columns.isin(['label'])].to_numpy()
            # print(f"\n\n {X_test_left_out.shape} \n\n")
            left_out_test_acc = model.sem_test(X_test_left_out, y_test_left_out)
            left_out_acc.append(left_out_test_acc)

        else:
            left_out_acc.append(0.)
        """

        # Train the knn model
        df_train_ = df_train.loc[df_train["label"].isin(known_labels), :]
        y_train = df_train_["label"].to_numpy()
        X_train = df_train_.loc[:, ~df_train_.columns.isin(['label'])].to_numpy()

        knn = KNeighborsClassifier(15, weights="uniform").fit(X_train, y_train)

        y_hat = knn.predict(X_test)
        correct_classifications = [1 for y_actual, y_pred in zip(y_test, y_hat) if y_actual == y_pred]
        knn_acc.append(len(correct_classifications) / y_test.shape[0])

    # Plot as barplot
    x_axis = [0, 1, 2, 3, 4]

    # print(left_out_acc)

    plt.bar(np.array(x_axis) - 0.2, np.array(sem_acc) * 100, width=0.2, align='center',
            label='Modified SVM acc using semantic vectors')
    plt.bar(np.array(x_axis) + 0.0, np.array(knn_acc) * 100, width=0.2, align='center',
            label='k-Nearest Neighbor acc')
    plt.bar(np.array(x_axis) + 0.2, np.array(cen_acc) * 100, width=0.2, align='center',
            label='Modified SVM acc using centerpoints')
    # plt.bar(np.array(x_axis) + 0.4, np.array(left_out_acc) * 100, width=0.2, align='center',
    #         label='Modified SVM acc using semantic vectors and only testing with left out labels')

    plt.xticks(x_axis, x_axis)
    plt.xlabel("Number of labels left out of the training data")
    plt.ylabel("Prediction accuracy (%)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    freeze_support()
    main2()

