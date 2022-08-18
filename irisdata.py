from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
import mglearn
import warnings

warnings.filterwarnings("ignore")

# Comon dataset which is iris dataset of sklearn
iris_dataset = load_iris()

# Printing keys of the dataset
# Note that keys are dictionary_keys it must be converted to list

print("This is keys of the data_set:{}\n".format(list(iris_dataset.keys())))
"""
OUTPUT:This is keys of the data_set:['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']
DESCR is basically description of the dataset.
"""
# Let's print description of the datasets.
print("This is description of dataset:{}".format(iris_dataset["DESCR"]))
print("This is target names of dataset:{}".format(iris_dataset["target_names"]))
print("This is feature names of dataset:{}".format(iris_dataset["feature_names"]))
print("This is type of data:{}".format(type(iris_dataset.get("data"))))  # Shows that iris data set is numpy array.
print("Shape of the data:{}".format(iris_dataset.get("data").shape))

# Shape of the data:(150, 4) 150 kolon asagı dogru 4 satır yana dogru. Yani 4 baslıga dair 150 veri var.

print("This is target of labeling species:{}".format(iris_dataset.get("target")))

# 0 means satosa,1 means versicolor,2 means virginica

print("This is target of labeling species length:{}".format(iris_dataset.get("target").shape[0]))

# Data test splitting for the learning and training.

X_train, X_test, y_train, y_test = train_test_split(iris_dataset.get("data"), iris_dataset.get("target"),
                                                    random_state=0)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.get("feature_names"))
# This is creating PANDAS DATAFRAME!!!!

print(iris_dataframe)

# DATA VISUALISATION

plt.figure()
sepal_length = plt.plot(iris_dataframe.get("sepal length (cm)"), label='sepal length (cm)')
sepal_width = plt.plot(iris_dataframe.get("sepal width (cm)"), label='sepal width (cm)')
petal_length = plt.plot(iris_dataframe.get("petal length (cm)"), label='petal length (cm)')
petal_width = plt.plot(iris_dataframe.get("petal width (cm)"), label='petal width (cm)')

# Printing the averages-means of the data set.
sepal_length_avg = np.full([1, iris_dataframe.get("sepal length (cm)").shape[0]],
                           iris_dataframe.get("sepal length (cm)").mean())
sepal_width_avg = np.full([1, iris_dataframe.get("sepal width (cm)").shape[0]],
                          iris_dataframe.get("sepal width (cm)").mean())
petal_length_avg = np.full([1, iris_dataframe.get("petal length (cm)").shape[0]],
                           iris_dataframe.get("petal length (cm)").mean())
petal_width_avg = np.full([1, iris_dataframe.get("petal width (cm)").shape[0]],
                          iris_dataframe.get("petal width (cm)").mean())

# Printing the averages-means of the data set for visuallize.
plt.plot(sepal_length_avg[0])
plt.plot(sepal_width_avg[0])
plt.plot(petal_length_avg[0])
plt.plot(petal_width_avg[0])

plt.legend(["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "sepal_length_avg",
            "sepal_width_avg", "petal_length_avg", "petal_width_avg"])
plt.show()
counter = 0
plt.figure()
# Printing the points of the data set.

sepal_length2 = plt.scatter(iris_dataframe.get("sepal length (cm)"), y_train)
sepal_width2 = plt.scatter(iris_dataframe.get("sepal width (cm)"), y_train)
petal_length2 = plt.scatter(iris_dataframe.get("petal length (cm)"), y_train)
petal_width2 = plt.scatter(iris_dataframe.get("petal width (cm)"), y_train)
plt.legend(["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])
plt.show()
plt.figure()
# Printing density map of the y_train.
plt.hist(y_train)
plt.show()
plt.figure()
# Printing 2d histogram.
plt.hist2d(iris_dataframe.get("sepal length (cm)"), y_train)
plt.hist2d(iris_dataframe.get("sepal width (cm)"), y_train)
plt.hist2d(iris_dataframe.get("petal length (cm)"), y_train)
plt.hist2d(iris_dataframe.get("petal width (cm)"), y_train)

plt.show()
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker="0", hist_kwds={'bins': 20}, s=60, alpha=.8,
                     cmap=mglearn.cm3)
knn_algo = KNeighborsClassifier(n_neighbors=1)
knn_algo.fit(X_train, y_train)
prediction = knn_algo.predict(X_test)
print("Predictions are {}".format(prediction))
print("Real Values are {}".format(y_test))

accuracy = knn_algo.score(X_test, y_test)
print("This is experiment accuracy:%{:.0f}".format(accuracy * 100))



# End Of Code
"""
<hr>
2022@CANROLLAS | CANROLLAS@gmail.com
<hr>
"""
