# Numpy implementation of KNN, adapted from work completed in CS321n
# Using CIFAR-10 from http://www.cs.toronto.edu/~kriz/cifar.html

import numpy as np
import time
import sys

import KNN

try:
    from data_utils import load_CIFAR10
except ImportError:
    print '[!] data_utils.py not found '
    print '[!] Download http://cs231n.stanford.edu/assignments/2016/winter1516_assignment1.zip and run get_datasets.sh'
# data_utils can be found in CS321N Github - https://github.com/cs231n/cs231n.github.io


def main(fulldata=False):

    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    num_training = 5000
    num_test = 500

    if fulldata:
        num_training = X_train.shape[0]

    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    if fulldata:
        num_test = X_test.shape[0]

    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    model = KNN.KNearestNeighbor()
    print "[+] Training Model using %d examples" % num_training
    model.train(X_train, y_train)

    print '[+] Calculating L2 Distnace for %d test cases' % num_test
    t1 = time.time()
    distance = model.L2_distance(X_test)
    print '[=] Distances Calulated'
    print '[=] Time taken: ', str(time.time() - t1)
    predict_y = model.predict(distance, 10)

    num_correct = np.sum(predict_y == y_test)
    accuracy = float(num_correct) / X_test.shape[0]
    print '[=] Got %d / %d correct => accuracy: %f' % (num_correct, X_test.shape[0], accuracy)

if __name__ == '__main__':
    print '[+] Example usage of KNN.py'

    try:
        if sys.argv[1] == 'full':
            print '[!] Using full dataset, this will be slow'
            main(fulldata=True)
    except IndexError:
        pass

    print '[!] Using sample dataset for speed.\n[!] To use full dataset type: python ExampleUsage.py full'
    main()
