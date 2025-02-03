import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 📂 Caricare le predizioni dal JSON
predictions_df = pd.read_csv('/Users/annamarika/Desktop/ensemble_predictions.csv')

# 📂 Caricare il dataset di test con le etichette reali
test_df = pd.read_csv('/Users/annamarika/PycharmProjects/VisualFoodRecognitionProject/ground_truth/filtered_images.csv',
                      header=None, names=['image', 'label'])

# 📌 Creare un dizionario {image_id: label reale} per accesso rapido
true_labels = dict(zip(test_df['image'], test_df['label']))

# 🔎 Estrarre le etichette vere e predette
y_true = []
y_pred = []

for _, row in predictions_df.iterrows():
    image_id = row['image_id']
    predicted_label = row['predicted_label']
    if image_id in true_labels:  # Controllo che l'immagine sia presente nel dataset
        y_true.append(true_labels[image_id])
        y_pred.append(predicted_label)


# ✅ Calcolare l'accuracy
accuracy = accuracy_score(y_true, y_pred)

# 📊 Generare il classification report
class_report = classification_report(y_true, y_pred, digits=4)

# 🔢 Calcolare la matrice di confusione
conf_matrix = confusion_matrix(y_true, y_pred)

# 📢 Stampare i risultati
print(f'Accuracy: {accuracy:.4f}')
print('\nClassification Report:\n', class_report)
print('\nConfusion Matrix:\n', conf_matrix)

# 📝 Salvare i risultati su file
with open('/Users/annamarika/Desktop/classification_report.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy:.4f}\n\n')
    f.write('Classification Report:\n')
    f.write(class_report)
    f.write('\nConfusion Matrix:\n')
    np.savetxt(f, conf_matrix, fmt='%d')
