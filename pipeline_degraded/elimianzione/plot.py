import matplotlib.pyplot as plt
import pandas as pd

from scripts.ImageDataset import ImageDataset

test_df = pd.read_csv('/Users/annamarika/PycharmProjects/VisualFoodRecognitionProject/ground_truth/my_val_info.csv', header=None, names=['image', 'label'])
test_dataset = ImageDataset(test_df, '/Users/annamarika/Desktop/val_set_degraded', train=False, deprecated=False)
predictions_df = pd.read_csv('/Users/annamarika/Desktop/ensemble_predictions.csv')

food_df = pd.read_csv('/Users/annamarika/PycharmProjects/VisualFoodRecognitionProject/ground_truth/foods_names.csv')  # Modifica il percorso se necessario
ltf = dict(zip(food_df['Index'], food_df['Food']))

merged_df = test_df.merge(predictions_df, left_on='image', right_on='image_id')

# Filtra solo le immagini classificate correttamente
# correct_predictions = merged_df[merged_df['label'] == merged_df['predicted_label']]
incorrect_predictions = merged_df[merged_df['label'] != merged_df['predicted_label']]

plt.figure(figsize=(17, 15))
num_images = min(16, len(incorrect_predictions))  # Mostriamo al massimo 16 immagini
for j, (_, row) in enumerate(incorrect_predictions.iterrows()):
    if j >= 16:
        break  # Limita a 16 immagini

    image = test_dataset.get_image_by_id(row['image'])  # Ottieni immagine
    true_label = ltf[row['label']]
    predicted_label = ltf[row['predicted_label']]
    confidence = row['confidence']

    plt.subplot(4, 4, j + 1)
    plt.imshow(image)
    plt.title(f'True Label: {true_label}\nPredicted: {predicted_label} ({confidence:.2f})')
    plt.axis('off')

plt.show()