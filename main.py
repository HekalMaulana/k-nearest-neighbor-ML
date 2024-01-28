import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Import Dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training K-Nearear Neighbor Model
classified = KNeighborsClassifier()
classified.fit(X_train, y_train)

print(classified.predict([[32, 150000]]))


