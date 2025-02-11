import pandas as pd
import ydata_profiling
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import fbeta_score

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("..\\dataset\\heart.csv")
df.head()

#cancello duplicati
df = df.drop_duplicates()

X = df.drop('output', axis = 1)
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, shuffle=True, random_state = 42)



# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


print("KNN Classification :")
print(classification_report(y_test, y_pred_knn))

print("-" * 40) 

f2_knn = fbeta_score(y_test, y_pred_knn, beta=2)
print(f"F2 Score: {f2_knn}\n")

print("-" * 40) 


# DECISION TREE CLASSIFIER
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)


print("Decision Tree Classification :")
print(classification_report(y_test, y_pred_dtc))

print("-" * 40) 

f2_dtc = fbeta_score(y_test, y_pred_dtc, beta=2)
print(f"F2 Score: {f2_dtc}\n")

print("-" * 40) 


# RANDOM FOREST CLASSIFIER
rd_clf = RandomForestClassifier()
rd_clf.fit(X_train, y_train)
y_pred_rd_clf = rd_clf.predict(X_test)


print("Random Forest Classification :")
print(classification_report(y_test, y_pred_rd_clf))

print("-" * 40)  

f2_rd_clf = fbeta_score(y_test, y_pred_rd_clf, beta=2)
print(f"F2 Score: {f2_rd_clf}")

print("-" * 40)  


base_classifier = DecisionTreeClassifier(max_depth=1)

ada = AdaBoostClassifier(base_classifier, n_estimators=50)  
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)


print("Ada Boost Classifier Classification :")
print(classification_report(y_test, y_pred_ada))

print("-" * 40) 

f2_ada = fbeta_score(y_test, y_pred_ada, beta=2)
print(f"F2 Score: {f2_ada}")

print("-" * 40)  


xgb = xgb.XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)


print("XGBoost Classifier Classification :")
print(classification_report(y_test, y_pred_ada))

print("-" * 40) 

f2_ada = fbeta_score(y_test, y_pred_ada, beta=2)
print(f"F2 Score: {f2_ada}")

print("-" * 40) 


