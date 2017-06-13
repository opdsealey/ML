# Numpy implementation of KNN, adapted from work completed in CS321n

import numpy as np

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def auto(self, X, k):
        distance = self.L2_distance(X)
        return self.predict(k, distance)

    # x: numpy array of ndshape, (N x D) such that N is number of training examples each with D dimensions
    # y: correct labels of (N,) such that y[i] = c where  c is the correct label for x[i]
    def train(self, x,y):
        self.x_train = x
        self.y_train = y

    # Returns array of L2 distances of X to X_train
    def L2_distance(self, X):

        # Theory - Compute L2 Distance
        # Using (x-y)^2 = x^2 + y^2 - 2xy
        # Distance = sqrt(x^2 + y^2 - 2xy)

        N_test = X.shape[0]
        N_train = self.x_train.shape[0]

        distance = np.zeros((N_test, N_train))

        # 2xy
        txy = -2 * np.dot(X, self.x_train.T)

        # X^2 + Y^2 , using boradcast to create correctly sized array
        sqrs = np.sum(X ** 2, axis=1)[:, np.newaxis] + np.sum(self.x_train ** 2, axis=1)
        distance = np.sqrt(sqrs + txy)

        return distance

    # Returns the label for all given test points
    def predict(self, distance, k=1):

        # k: Number of neighbours to poll, currently using a vote based assessment
        # distance: ndarray such that distance[i,j] provides the distance between
        # the ith test point and the jth train point
        N_test = distance.shape[0]
        prediction = np.zeros(N_test)
        for i in xrange(N_test):
            indexes = np.argsort(distance[i, :])[0:k]
            closest_y = np.take(self.y_train, indexes)

            prediction[i] = np.argmax(np.bincount(closest_y))

        return prediction
