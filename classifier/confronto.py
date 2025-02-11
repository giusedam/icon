import matplotlib.pyplot as plt

modelli = ['Decision Tree', 'Random Forest', 'Ada Boost', 'XGBoost', 'kNN']

accuracies = [87, 90, 77, 77, 93]

plt.figure(figsize=(10, 6))
plt.barh(modelli, accuracies, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Confronto tra Modelli di Classificazione')
plt.xlim(0, 100)

for i, v in enumerate(accuracies):
    plt.text(v + 1, i, str(v) + '%', va='center', color='black', fontweight='bold')

plt.show()
