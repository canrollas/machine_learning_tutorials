import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from mglearn.datasets import load_extended_boston
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
X, y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lasso_regression = Lasso()
lasso_regression.fit(X_train, y_train)

print("Train score:{:2f}".format(lasso_regression.score(X_train, y_train)))
print("Test score:{:2f}".format(lasso_regression.score(X_test, y_test)))
print("Number of features used:{}".format(np.sum(lasso_regression.coef_ != 0)))
"""
---------------2022@CANROLLAS | CANROLLAS@gmail.com---------------
"""

lasso_regression1 = Lasso(alpha=0.01, max_iter=1000)
lasso_regression1.fit(X_train, y_train)
print("Train score:{:2f}".format(lasso_regression1.score(X_train, y_train)))
print("Test score:{:2f}".format(lasso_regression1.score(X_test, y_test)))
print("Number of features used:{}".format(np.sum(lasso_regression1.coef_ != 0)))
plt.figure()
test_scores = np.array([])
train_scores = np.array([])
alpha_x = np.array([])
for general in range(1, 10000):
    lasso_regression2 = Lasso(alpha=general / 10000, max_iter=10000)
    lasso_regression2.fit(X_train, y_train)
    test_score = lasso_regression2.score(X_test, y_test)
    train_score = lasso_regression2.score(X_test, y_test)
    test_scores = np.append(test_scores, [test_score])
    train_scores = np.append(train_scores, [train_score])
    alpha_x = np.append(alpha_x, [general / 10000])

plt.plot(test_scores, alpha_x, label="Test Score")
plt.xlabel("Alpha value")
plt.ylabel("Score value")
plt.legend()
plt.title("Score Measure Chart")
plt.show()

plt.plot(train_scores, alpha_x, label="Train score")
plt.xlabel("Alpha value")
plt.ylabel("Score value")
plt.legend()
plt.title("Score Measure Chart")
plt.show()
