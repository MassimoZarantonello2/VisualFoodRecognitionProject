import os
import numpy as np
import pandas as pd
from models.FoodCNN import FoodCNN
from scripts.ImageDataset import ImageDataset
import time

def train_models(model_list, data_dataset, prediction_loader, cycle, num_epochs, lc):
    models_accuracies = []
    models_predictions = []
    for model in model_list:
        model = FoodCNN(model_name=model)
        path = f'./models/trained_models/{model.model_name}_{cycle}.pth'
        if os.path.exists(path):
            model.load_model(path)
            print(f'Loading model {model.model_name}_{cycle}.pth')
            lc.write(f'     Loading model {model.model_name}_{cycle}.pth')
            models_accuracies.append(-1)
        else:
            print(f'Training model {model.model_name}')
            lc.write(f'Training model {model.model_name}')
            start = time.time()
            train_loss, val_loss, train_accuracy, val_accuracy = model.train_model(data_dataset, validation=0.1, num_epochs=num_epochs, cycle = cycle)
            end = time.time()
            lc.write(f'     Training time: {end - start}')
            models_accuracies.append(val_accuracy[4])
        start = time.time()
        predictions = model.predict(prediction_loader)
        end = time.time()
        lc.write(f'     Prediction time: {end - start}')
        models_predictions.append(predictions)
        lc.write(f'     Model {model.model_name} accuracy: {models_accuracies[-1]}')
    return models_accuracies, models_predictions

def get_agreement_and_treshold(predictions, iterative_train_dataframe, weights, confidence, lc):
    '''
    ### input:
    predictions: list of predictions propability for each model 
    weights: list of weights for each model
    confidence: threshold for the agreement
    ### return: 
    list of images index
    list of images label
    '''
    res_predictions, eff_predictions, vgg_predictions = predictions
    res_weight, eff_weight, vgg_weight = weights
    images_idx = []
    images_label = []
    images_to_add = 0
    for i in range(len(res_predictions)):
        res_label = int(np.argmax(res_predictions[i]))
        eff_label = int(np.argmax(eff_predictions[i]))
        vgg_label = int(np.argmax(vgg_predictions[i]))

        res_confidence = res_predictions[i][res_label]
        eff_confidence = eff_predictions[i][eff_label]
        vgg_confidence = vgg_predictions[i][vgg_label]

        if res_label == eff_label and res_label == vgg_label:
            models_confidence = res_confidence * res_weight + eff_confidence * eff_weight + vgg_confidence * vgg_weight
            if models_confidence > confidence:
                images_to_add += 1
                images_idx.append(iterative_train_dataframe.get_image_id(i))
                images_label.append(res_label)

    print(f'Images to add: {images_to_add}')
    return images_idx, images_label

def update_datasets(images_idx, images_label, unlabeld_train_dataset, iterative_train_dataset):
    '''
    images_idx: list of names of the images to add
    images_label: list of labels of the images to add
    iterative_train_dataset: dataset to add the images
    unlabeld_train_dataset: dataset to remove the images
    '''
    for image_id, image_label in zip(images_idx, images_label):
        iterative_train_dataset.add_image(image_id, image_label)
        unlabeld_train_dataset.remove_image(image_id)

    return iterative_train_dataset, unlabeld_train_dataset