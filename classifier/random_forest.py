import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
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

rfc = RandomForestClassifier(
    criterion='entropy',
    max_depth=2,
    min_samples_leaf=25,
    min_samples_split=25,
    n_estimators=50,
    random_state=42
)

rfc.fit(X_train, y_train)


accuracy = cross_val_score(rfc, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-validation accuracy:", accuracy)
print("\nMean accuracy:", accuracy.mean(), "\nStd deviation:", accuracy.std())

y_pred = rfc.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", test_accuracy)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8, 5))
cm = confusion_matrix(y_test, y_pred)
plt.title('Confusion Matrix', fontweight='bold')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.arange(2), ['No', 'Yes'])
plt.yticks(np.arange(2), ['No', 'Yes'])
plt.show()

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
predictions = rfc.predict(new_data)

for i, prediction in enumerate(predictions):
    risultato = "puo avere un attacco di cuore" if prediction == 1 else "puo avere un attacco di cuore"
    print(f"Esempio {i+1}: Il modello predice che la persona {risultato}.")

if hasattr(rfc, "predict_proba"): 
    probabilities = rfc.predict_proba(new_data)
    print("\nProbabilità delle classi per i nuovi dati:")
    print(probabilities)