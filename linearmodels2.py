from mglearn.datasets import  load_extended_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X,y = load_extended_boston()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

lr = LinearRegression()
lr.fit(X_train,y_train)
print("Accuracy of the test {}".format(lr.score(X_test,y_test)))

