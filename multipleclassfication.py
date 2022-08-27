import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
print(X)  # classfication coordinates.
print(y)  # classficiation point.
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("Classification Problem")
support_vector = LinearSVC()
support_vector.fit(X, y)
print("Lets see coefficient shape and intercept of vectors")
print("Coefficient:{}".format(support_vector.coef_.shape))
print("Intercept:{}".format(support_vector.intercept_.shape))
# 3 tane vektor var ve iki özellige sahip.
# 3 vektorden 3 ü kesişim yapıyor.
line = np.linspace(-15, 15)
for coef, intercept, color in zip(support_vector.coef_, support_vector.intercept_, ["b", "r", "g"]):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color,label=color)
plt.legend(["Class 0", "Class 1", "Class 2","Blue Line","Red Line","Yellow Line"],loc='upper left')
plt.show()
