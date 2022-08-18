import mglearn
import mglearn.datasets as datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = datasets.make_forge()
# Random state 0 => It will be same.
# Random state None => Shuffle every time.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train, y_train)

# Prediction
prediction = classifier.predict(X_test)
print("This is prediction:{}".format(prediction))
print("This are real values:{}".format(y_test))
print("Accuracy is {:.2f}".format(classifier.score(X_test, y_test)))

for n_neighbours in [1,3,9]:
    plt.figure()
    clf2 = KNeighborsClassifier(n_neighbors=n_neighbours).fit(X_train,y_train)
    mglearn.plots.plot_2d_separator(clf2,X,fill=True,eps=0.5,alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y)
    plt.ylabel("Feature-0")
    plt.ylabel("Feature-1")
    plt.title("{}th closest neighbour".format(n_neighbours))
    print("Accuracy is according to neighbour {:.2f}".format(clf2.score(X_test, y_test)))
    plt.legend()
    plt.show()