import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression,Ridge

le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

data = pd.read_csv("veri.csv")


"""
Buradaki temel amac ilk basta boy ile cinsiyet arasındaki iliskiyi tutturmak
MILESTONE: BOY VE CINSIYET
"""
boy = data.iloc[:, 1].values
boy = np.array(boy).reshape(-1,1)
gender_label = data.iloc[:, 4]
numeric_gender_label = le.fit_transform(gender_label)
numeric_gender_label = np.array(numeric_gender_label).reshape(-1,1)

print(boy,"\n",numeric_gender_label)

X_train,X_test,y_train,y_test = train_test_split(boy,numeric_gender_label,random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)
y_predict = linear_model.predict(X_test)
print("Accuracy is {:.2f}".format(linear_model.score(X_test,y_test)))

# Ozür dilerim sadi evren seker verdigin veri seti berbat maalesef...
