# Logistic Regression Model

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import ML libaries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# load dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# feature scaling the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# training logistic regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# predicting a new result
classifier.predict(sc.transform([[30, 87000]]))
print(classifier)

# predicting test set results
y_pred = classifier.predict(X_test)
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
)
