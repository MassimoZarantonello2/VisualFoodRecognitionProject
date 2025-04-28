from models.EnsambleModel import EnsambleModel
import pandas as pd
from scripts.ImageDataset import ImageDataset


models_names = ['resnet18', 'efficientnet','vgg16']
models_weights = [0.32075472, 0.33692722, 0.34231806]

test_df = pd.read_csv('/ground_truth/filtered_images.csv', header=None, names=['image', 'label'])
test_dataset = ImageDataset(test_df, '../Desktop/new_val', train=False, deprecated=False)

em = EnsambleModel(models_name=models_names, pre_trained=True, models_weights=models_weights)
images_idx, images_label, predictions_confidences = em.predict(test_dataset)

results_df = pd.DataFrame({
    'image_id': images_idx,
    'predicted_label': images_label,
    'confidence': predictions_confidences
})

results_df.to_csv('../ensemble_predictions.csv', index=False)