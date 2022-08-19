import matplotlib.pyplot as plt
import mglearn.datasets
from mglearn.plots import plot_linear_regression_wave
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

plot_linear_regression_wave()
plt.show()

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
print("Coeefficients:{}".format(format(linear_regression.coef_[0])))
print("Intercept:{}".format(format(linear_regression.intercept_)))
print("Accuracy of the test is:{}".format(linear_regression.score(X_test, y_test)))
