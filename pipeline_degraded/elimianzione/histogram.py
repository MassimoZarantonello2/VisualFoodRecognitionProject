import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Carica il dataset di ground truth
test_df = pd.read_csv('/Users/annamarika/PycharmProjects/VisualFoodRecognitionProject/ground_truth/my_val_info.csv', header=None, names=['image', 'label'])

# Carica le predizioni
predictions_df = pd.read_csv('/Users/annamarika/Desktop/ensemble_predictions.csv')  # Modifica il percorso se necessario

# Merge tra ground truth e predizioni
merged_df = test_df.merge(predictions_df, left_on='image', right_on='image_id')

# Aggiunge una colonna per indicare se la predizione è corretta
merged_df['correct'] = merged_df['label'] == merged_df['predicted_label']

# Calcola l'accuratezza per ogni classe
accuracy_per_class = merged_df.groupby('label')['correct'].mean().reset_index()

# Imposta lo stile di Seaborn
sns.set_style("whitegrid")

# Crea il grafico a barre con Seaborn
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=accuracy_per_class, x='label', y='correct', color="blue", alpha=0.7)

# Personalizzazione del grafico
plt.xlabel('Class (Label Number)')
plt.ylabel('Accuracy')
plt.title('Accuracy per Class')

# Mostra le etichette solo ogni 50 classi
labels = accuracy_per_class['label']
ax.set_xticks(labels[::50])
ax.set_xticklabels(labels[::50], rotation=90)  # Ruota le etichette per leggibilità

plt.show()
