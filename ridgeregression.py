import matplotlib.pyplot as plt
import mglearn.plots
from mglearn.datasets import load_extended_boston
from sklearn.model_selection import train_test_split
import warnings
from sklearn.linear_model import Ridge, LinearRegression

warnings.filterwarnings("ignore")
ridge = Ridge()
X, y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge.fit(X_train, y_train)
print("Training score is %{:.2f} ".format(ridge.score(X_train, y_train) * 100))
print("Test score is %{:.2f} ".format(ridge.score(X_test, y_test) * 100))
print("------------------------------------------------------------------------")
# Changing the alpha
ridge10 = Ridge(alpha=10)
ridge10.fit(X_train,y_train)
print("Training score is %{:.2f} ".format(ridge10.score(X_train, y_train) * 100))
print("Test score is %{:.2f} ".format(ridge10.score(X_test, y_test) * 100))
# -------------------------
ridge01 = Ridge(alpha=0.1)
ridge01.fit(X_train,y_train)
print("Training score is %{:.2f} ".format(ridge01.score(X_train, y_train) * 100))
print("Test score is %{:.2f} ".format(ridge01.score(X_test, y_test) * 100))
#***************************
X,y = load_extended_boston()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

lr = LinearRegression()
lr.fit(X_train,y_train)
#***************************
# Plotting the data
plt.plot(ridge.coef_,"s",label="Ridge alpha=1")
plt.plot(ridge10.coef_,"^",label="Ridge alpha=10")
plt.plot(ridge01.coef_,"v",label="Ridge alpha=0.1")
plt.plot(lr.coef_,"o",label="LinearRegression")
plt.xlabel("Coefficient Index")
plt.ylabel("Coefficient Magnitude")
plt.ylim(-25,25)
plt.legend()
plt.show()
mglearn.plots.plot_ridge_n_samples()
plt.show()


