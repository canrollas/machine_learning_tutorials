from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import pandas as pd
label_encoder = preprocessing.LabelEncoder()
dataframe = pd.read_csv("golf-dataset.csv")
print(dataframe.keys())
golf_result = pd.DataFrame(label_encoder.fit_transform(dataframe["Play Golf"]),columns=["Play Golf"])
outlook = pd.DataFrame(label_encoder.fit_transform(dataframe["Outlook"]),columns=["Outlook"])
temp = pd.DataFrame(label_encoder.fit_transform(dataframe["Temp"]),columns=["Temp"])
humidity = pd.DataFrame(label_encoder.fit_transform(dataframe["Humidity"]),columns=["Humidity"])
windy = pd.DataFrame(label_encoder.fit_transform(dataframe["Windy"]),columns=["Windy"])

composed_data_frame = pd.concat([outlook,temp,humidity,windy,golf_result],axis=1)
print(composed_data_frame)

gaussianClassifier = GaussianNB()
X_train,X_test,y_train,y_test = train_test_split(composed_data_frame.iloc[:,0:4],composed_data_frame.iloc[:,4],random_state=42)
gaussianClassifier.fit(X_train,y_train)
print("This is accuracy of the test gaussian: %{:.2f}".format(gaussianClassifier.score(X_test,y_test)*100))

bernoulli_dist = BernoulliNB()
bernoulli_dist.fit(X_train,y_train)
print("This is accuracy of the test bernoulli :%{:.2f}".format(bernoulli_dist.score(X_test,y_test)*100))

multinomial = MultinomialNB()
multinomial.fit(X_train,y_train)
print("This is accuracy of the test Multinomial :%{:.2f}".format(multinomial.score(X_test,y_test)*100))


