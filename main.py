import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

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

# Predicting a new training set result
print(classified.predict([[32, 150000]]))

# Predicting a Test Set Result
y_pred = classified.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making a confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))


