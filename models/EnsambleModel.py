import torch
from .FoodCNN import FoodCNN
from scripts.ImageDataset import ImageDataset
from utils.train_support import *
import numpy as np

class EnsambleModel():

    def __init__(self, models_name, models_weights, num_classes=251):
        self.models_name = models_name
        self.models_weights = models_weights
        self.num_classes = num_classes
        self.models = []
        self.models = self.get_models()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_ensamble(self, train_dataset, lr, num_epochs=10, lc=None):
        for model in self.models_name:
            model = FoodCNN(model_name=model, num_classes=self.num_classes)
            _, _, _, val_accuracies = model.train_model(train_dataset, validation=0.2, lr=lr, num_epochs=num_epochs, cycle=-1, lc=lc)
            self.models.append(model)
            self.models_weights.append(val_accuracies[-1])

    def get_models(self):
        for model_name in self.models_name:
            model = FoodCNN(model_name)
            print('Loading model: ', model_name)
            model.load_model('./models/trained_models/' + model_name + '_14.pth')
            self.models.append(model)
        return self.models
    

    def predict(self, image_dataset, lc=None):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(image_dataset))

        res_predictions, eff_predictions, vgg_predictions = predictions
        res_weight, eff_weight, vgg_weight = self.models_weights
        images_idx = []
        images_label = []
        predictions_confidences = []
        images_to_add = 0
        for i in range(len(res_predictions)):
            weighted_prediction = res_weight * res_predictions[i] + eff_weight * eff_predictions[i] + vgg_weight * vgg_predictions[i]
            max_prob = np.max(weighted_prediction)
            most_probable_class = np.argmax(weighted_prediction)
            predictions_confidences.append(max_prob)
            images_idx.append(image_dataset.get_image_id(i))
            images_label.append(most_probable_class)
            
        return images_idx, images_label, predictions_confidences
