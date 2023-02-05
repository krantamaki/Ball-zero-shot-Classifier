import numpy as np
from multiprocessing import freeze_support
from timeit import default_timer as timer
from datetime import timedelta
from ellipse_classifier import EllipseClassifier
from visualize import viz_ellipse_classifier, viz_classic_neural_net, viz_knearest_neighbors
from ball_classifier import BallClassifier
from leave_n_out_split import leave_n_out_split


def gen_points(mean, deviation, n):
    x = np.random.normal(mean[0], deviation[0], size=n)
    y = np.random.normal(mean[1], deviation[1], size=n)
    return np.column_stack((x, y))


def gen_sem_vector(mean):
    return np.array([mean[0] ** 2 - mean[0], mean[0] ** (1 / 2) + mean[1] ** (3 / 2),
                     2 * mean[1] - mean[0] ** 2, mean[1] ** 3])
    # return np.array(mean)


def main():
    # Means and standard deviations of given labels
    means = [(1., 1.), (1., 2.), (1.5, 1.5), (2., 2.), (0.5, 1.), (0.5, 0.5), (3., 1.), (1.5, 0.5), (3., 2.), (2., 1.),
             (0.5, 2.), (1., 0.5), (2., 0.5), (2.5, 2.5), (1., 1.5), (1.5, 1.), (1.5, 2.5), (2.5, 1.5), (1., 0.), (2.5, 0.),
             (0., 1.), (2., 0.), (3.5, 0), (0.5, 2.5), (3., 1.5), (3., 0.5), (2, 2.5), (0., 2.5), (3., 0.), (3.5, 1.5)]

    deviations = [(0.2, 0.1), (0.3, 0.2), (0.3, 0.2), (0.3, 0.1), (0.2, 0.3), (0.2, 0.2), (0.2, 0.25), (0.2, 0.3), (.4, 0.2), (0.15, 0.2),
                  (0.2, 0.1), (0.15, 0.15), (0.1, 0.3), (0.1, 0.2), (0.3, 0.2), (0.1, 0.1), (0.15, 0.2), (0.2, 0.2), (0.15, 0.25), (0.3, 0.1),
                  (0.2, 0.2), (0.1, 0.1), (0.3, 0.3), (0.2, 0.1), (0.3, 0.05), (0.2, 0.1), (0.2, 0.1), (0.3, 0.2), (0.1, 0.1), (0.1, 0.1)]

    #means = [(1., 1.), (1., 2.), (1.5, 1.5), (2., 2.), (0.5, 1.), (0.5, 0.5), (3., 1.), (1.5, 0.5), (3., 2.), (2., 1.)]
    #deviations = [(0.2, 0.1), (0.3, 0.2), (0.15, 0.3), (0.3, 0.2), (0.2, 0.3), (0.2, 0.4), (0.2, 0.25), (0.3, 0.1), (.4, 0.2), (0.15, 0.4)]

    # How many points are wanted per label
    num_points = 15
    # How many labels are used
    num_labels = 10

    X = np.concatenate([gen_points(means[i], deviations[i], num_points) for i in range(num_labels)])
    y = np.concatenate([[f"$\mu$: {means[i]}, $\sigma^2$: {deviations[i]}"] * num_points for i in range(num_labels)])

    X_train, y_train, X_test, y_test = leave_n_out_split(X, y, 3, test_only_with_excluded=False)
    print()

    model = EllipseClassifier(base_gamma=1, y_multiple=5)

    # Time the training of the model
    start = timer()
    model.par_train(X_train, y_train)
    end = timer()
    print(f"\n\nTime elapsed during training: {timedelta(seconds=end - start)}.\n")

    viz_ellipse_classifier(model, X_train, y_train, show=True, save_path="random_data.png")

    # Log the training accuracy
    print(f"Training accuracy: {model.eval()}\n")

    # Pass semantic information to the model
    S = np.array([gen_sem_vector(means[i]) for i in range(len(means))])
    sem_y = np.array([f"$\mu$: {means[i]}, $\sigma^2$: {deviations[i]}" for i in range(len(means))])
    model.add_sematic_vectors(S, sem_y)

    # Time the testing step
    start = timer()
    test_acc = model.sem_test(X_test, y_test)
    end = timer()
    print(f"\n\nTime elapsed during testing: {timedelta(seconds=end - start)}.")

    print(f"\n\nFinal testing accuracy: {test_acc}\n")

    # Visualize a MLP of shape 2 - 12 - 12 - 10
    # viz_classic_neural_net(y_train, X_train, [2, 12, 12, 10], np.unique(y_train), show=True, y_test=y_test, X_test=X_test)

    # Visualize a K-Nearest Neighbor model
    viz_knearest_neighbors(y_train, X_train, np.unique(y_train), 10, show=True, y_test=y_test, X_test=X_test)


if __name__ == '__main__':
    freeze_support()
    main()
