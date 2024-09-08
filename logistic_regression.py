# Logistic Regression Model

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import ML libaries
from sklearn.model_selection import train_test_split

# load dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
