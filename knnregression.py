import matplotlib.pyplot as plt
import numpy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings("ignore")


boston = load_boston()

regressor = KNeighborsRegressor()


def vissualize(dataset_X, target, unkown_x, unknown_y):
    X_target = numpy.array([])
    X_target2 = numpy.array([])

    for general in dataset_X:
        counter = 0
        for looper_iter in general:
            counter = counter + looper_iter
        X_target = numpy.append(X_target, [counter])
    for general in unkown_x:
        counter = 0
        for looper_iter in general:
            counter = counter + looper_iter
        X_target2 = numpy.append(X_target2, [counter])
    plt.figure()
    plt.scatter(X_target, target)
    plt.scatter(X_target2, unknown_y)
    plt.legend(["Train House price", "Test House price"])
    plt.xlabel("Sum of all parameters")
    plt.ylabel("Target of dataset")
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(boston.get("data"), boston.get("target"), random_state=0)
vissualize(X_train, y_train, X_test, y_test)
regressor.fit(X_train, y_train)
plt.show()
print("This is the score of the regressor:{}".format(regressor.score(X_test, y_test)))
