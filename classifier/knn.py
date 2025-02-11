import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import fbeta_score

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("..\\dataset\\heart.csv")
df = df.drop_duplicates()
scaler = StandardScaler()
df_scaler = df.copy()
num = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
scaled = scaler.fit_transform(df_scaler[num])
df_scaler[num] = scaled
X = df.drop('output', axis=1)
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=29)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

new_data = pd.DataFrame({
    "age": [63, 41],
    "sex": [1, 0],
    "cp": [3, 1],
    "trtbps": [145, 130],
    "chol": [233, 204],
    "fbs": [1, 0],
    "restecg": [0, 0],
    "thalachh": [150, 172],
    "exng": [0, 0],
    "oldpeak": [2.3, 1.4],
    "slp": [0, 2],
    "caa": [0, 0],
    "thall": [1, 2]
})

new_data[num] = scaler.transform(new_data[num])
predictions = knn.predict(new_data)

for i, prediction in enumerate(predictions):
    risultato = "puo avere un attacco di cuore" if prediction == 1 else "non puo avere un attacco di cuore"
    print(f"Esempio {i+1}: Il modello predice che la persona {risultato}.")

if hasattr(knn, "predict_proba"): 
    probabilities = knn.predict_proba(new_data)
    print("\nProbabilità delle classi per i nuovi dati:")
    print(probabilities)
