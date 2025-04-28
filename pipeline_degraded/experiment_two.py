from models.EnsambleModel import EnsambleModel
import pandas as pd
from scripts.ImageDataset import ImageDataset


models_names = ['resnet18', 'efficientnet','vgg16']


train_df = pd.read_csv('/ground_truth/my_iterative_train_augmented2.csv')
train_dataset = ImageDataset(train_df, '../train_set', train=True, deprecated=False)


test_df = pd.read_csv('../ground_truth/my_val_info.csv', header=None, names=['image', 'label'])
test_dataset = ImageDataset(test_df, '../improvement_degradated', train=False, deprecated=False)



em = EnsambleModel(models_name=models_names, pre_trained=False)

em.train_ensamble(train_dataset, num_epochs=10, lr=0.0001)
images_idx, images_label, predictions_confidences = em.predict(test_dataset)

results_df = pd.DataFrame({
    'image_id': images_idx,
    'predicted_label': images_label,
    'confidence': predictions_confidences
})

results_df.to_csv('../ensemble_predictions.csv', index=False)