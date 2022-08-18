import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
cancer = load_breast_cancer()

print("Shape of The data set => {} entry".format(cancer.get("data").shape))
# Test ve Train degerlerine esit sayıda aynı label dagıtmak icin stratify kullanılır.
# FIXME random state possible value'lara bak.
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                     random_state=66)
print("This is y_shape {}".format(y_test.shape[0]))
best_neighbour = 1
best = 0
plt.figure()
accuracy_of_test = np.array([])
accuracy_of_train = np.array([])

for neighbour in range(1,11):
    classifier = KNeighborsClassifier(n_neighbors=neighbour)
    classifier.fit(X_train,y_train)
    accuracy = classifier.score(X_test,y_test)*100
    accuracy_of_test = np.append(accuracy_of_test,[accuracy])
    accuracy_of_train = np.append(accuracy_of_train, [classifier.score(X_train,y_train)*100])
    plt.plot(accuracy_of_test)
    plt.plot(accuracy_of_train)

    if accuracy>=best:
        best = accuracy
        best_neighbour = neighbour
plt.legend(["Accuracy of Test","Accuracy of Train"])
plt.show()
print("This is best accuracy in {} neighbours which is {}".format(best_neighbour,best))


