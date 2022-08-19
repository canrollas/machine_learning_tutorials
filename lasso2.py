import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston

import warnings

warnings.filterwarnings("ignore")


X, y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge01 = Ridge(alpha=0.1)
ridge01.fit(X_train, y_train)

lasso_regression1 = Lasso(alpha=1, max_iter=1000)
lasso_regression1.fit(X_train, y_train)

lasso_regression01 = Lasso(alpha=0.01, max_iter=1000)
lasso_regression01.fit(X_train, y_train)

lasso_regression0001 = Lasso(alpha=0.0001, max_iter=1000)
lasso_regression0001.fit(X_train, y_train)

plt.figure()
plt.plot(ridge01.coef_, "^", label="Ridge 0.1")
plt.plot(lasso_regression0001.coef_, "v", label="Lasso 0.0001")
plt.plot(lasso_regression1.coef_, "s", label="Lasso 1")
plt.plot(lasso_regression01.coef_, "o", label="Lasso 0.1")
plt.xlabel("Coefficient Index")
plt.ylabel("Coefficient Magnitude")
plt.legend()
plt.show()
