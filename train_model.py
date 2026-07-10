import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle

#Load the dataset
df = pd.read_csv("diabetes.csv")

#top rows
print(df.head())

#bottom rows
print(df.tail())

#size of dataset
print("Size of the dataset:",df.shape)

print(df.info())

print(df.describe())
print("\n")

#NULL VALUES
print(df.isnull().sum())

#To count the number of 0s and 1s
print(df["Outcome"].value_counts())

#feature extraction
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

#splitting
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#model selection
model=LogisticRegression()

#model training
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#Evalution
print("accuracy Score:",accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#saving model
pickle.dump(model,open("diabetes_model.pkl","wb"))
pickle.dump(scaler,open("scaler.pkl","wb"))

print("\nModel and Scaler saved successfully.")