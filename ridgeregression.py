from mglearn.datasets import load_extended_boston
from sklearn.model_selection import train_test_split
import warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
ridge = Ridge()
X, y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge.fit(X_train, y_train)
print("Training score is %{:.2f} ".format(ridge.score(X_train, y_train) * 100))
print("Test score is %{:.2f} ".format(ridge.score(X_test, y_test) * 100))
print("------------------------------------------------------------------------")
# Changing the alpha
ridge01 = Ridge(alpha=0.1)
ridge01.fit(X_train,y_train)
print("Training score is %{:.2f} ".format(ridge01.score(X_train, y_train) * 100))
print("Test score is %{:.2f} ".format(ridge01.score(X_test, y_test) * 100))