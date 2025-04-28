import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

from scripts.ImageDataset import ImageDataset
from models.ResNet import ResNet
from models.SimpleCNN import SimpleCNN
from models.FoodCNN import FoodCNN

from models.CustomNet import CustomNet

from utils.print_errors import print_errors

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def train_models(model_list, data_dataset, prediction_loader, num_epochs, cycle):
    models_accuracies = []
    models_predictions = []
    for model in model_list:
        print(f'Training model {model}')
        model = FoodCNN(model_name=model)
        path = f'./models/trained_models/{model.model_name}_{cycle}.pth'
        if os.path.exists(path):
            model.load_model(path)
            print(f'Loading model {model.model_name}_{cycle}.pth')
            models_accuracies.append(-1)
        else:
            _,_,_, val_accuracy = model.train_model(data_dataset, validation=0.1, num_epochs=num_epochs, cycle = cycle)
            models_accuracies.append(val_accuracy[4])
        predictions = model.predict(prediction_loader)
        models_predictions.append(predictions)

    return models_accuracies, models_predictions

def get_agreement_and_treshold(predictions, weights, confidence):
    res_predictions, eff_predictions, vgg_predictions = predictions
    res_weight, eff_weight, vgg_weight = weights
    images_idx = []
    images_label = []
    images_to_add = 0
    for i in range(len(res_predictions)):
        res_label = np.argmax(res_predictions[i])
        eff_label = np.argmax(eff_predictions[i])
        vgg_label = np.argmax(vgg_predictions[i])

        res_confidence = res_predictions[i][res_label]
        eff_confidence = eff_predictions[i][eff_label]
        vgg_confidence = vgg_predictions[i][vgg_label]

        if res_label == eff_label and res_label == vgg_label:
            models_confidence = res_confidence * res_weight + eff_confidence * eff_weight + vgg_confidence * vgg_weight
            if models_confidence > confidence:
                images_to_add += 1
                images_idx.append(i)
                images_label.append(res_label)

    print(f'Images to add: {images_to_add}')
    return images_to_add, images_idx, images_label

def update_datasets(images_idx, images_label, predictions_dataset, train_df, unlabel_df):
    for i in range(len(images_idx)):
        # Ottengo l'immagine e la sua label dal dataset delle predizioni
        image_idx = images_idx[i]
        image_label = images_label[i]
        image_id = predictions_dataset.get_image_id(image_idx)
        
        # Aggiungo una nuova riga al DataFrame di train
        new_row = pd.DataFrame({'image_id': [image_id], 'label': [image_label]})
        train_df = pd.concat([train_df, new_row], ignore_index=True)
        
        # Rimuovo l'immagine dal dataset delle immagini senza label
        unlabel_df = unlabel_df[unlabel_df['image_id'] != image_id]

    return train_df, unlabel_df

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

ltf = pd.read_csv('./ground_truth/foods_names.csv')
ltf = ltf.drop(columns=['Index'])
label_to_foods = ltf.to_dict()['Food']

train_gt_path = './ground_truth/train_small.csv'
test_gt_path = './ground_truth/new_val_info.csv'
train_unlabel_path = './ground_truth/unlabeled.csv'

train_image_path = './train_set/'
train_df = pd.read_csv(train_gt_path, header=None, names=['image_id', 'label'])

test_image_path = './val_set/'
test_df = pd.read_csv(test_gt_path, header=None, names=['image_id', 'label'])
test_dataset = ImageDataset(test_df, test_image_path, train=False, labels = True)
test_dataset_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

batch_size = 32

unlabel_train_df = pd.read_csv(train_unlabel_path, header=None, names=['image_id'])
models_list =['resnet18', 'efficientnet','vgg16']


with open('utils/log.txt', 'w+') as f:
    for i in range(10):
        f.write(f'--------------------------------------\n')
        f.write(f'Inizio Iterazione {i}\n')
        # Dataset di training iterativo
        iterative_train_dataset = ImageDataset(train_df, train_image_path, train=True, labels = True)
        f.write(f'Numero immagini nel train set: {len(train_df)}\n')
        # Dataset commposto dalle immagini senza label che parte di cui verrà aggiunta al train successivo e rimossa da unlabel_train_dataset
        predictions_df = unlabel_train_df.sample(frac=0.05, random_state=42, replace=False)
        f.write(f'Numero immagini da predirre a cui assegnare label: {len(unlabel_train_df)}\n')
        predictions_dataset = ImageDataset(predictions_df, train_image_path, train=False, labels = False)
        predictions_loader = DataLoader(predictions_dataset, batch_size=batch_size, shuffle=False)

        # Prendo i modelli e li alleno con l'iterative train dataset
        traning_start = time.time()
        models_accuracies, models_predictions = train_models(models_list, iterative_train_dataset, predictions_loader, num_epochs=10, cycle = i)
        training_end = time.time()
        f.write(f'Tempo di training: {training_end - traning_start}')
        for model, a in zip(models_list,models_accuracies):
            f.write(f'{model} accuracy: {a}\n')

        # Calcolo i pesi dei modelli in base all'accuratezza sul validation set
        weights = np.array(models_accuracies) / np.sum(models_accuracies)
        for model, w in zip(models_list,weights):
            f.write(f'{model} weight: {w}\n')

        # gli output dei modelli sono dei vettori di probabilità, calcolo quando tutti e tre i modelli sono d'accordo e hanno una confidenza maggiore di 0.75
        number_of_images, images_index, images_label = get_agreement_and_treshold(models_predictions, weights, confidence=0.75) 
        f.write(f'Numero immagini da aggiungere al train set: {number_of_images}\n')
        # Prendo le immagini che hanno superato questo test, assegno loro le label le rimuovo dall' unlabel_set e le aggiungo all'iterative train dataset
        # Lunghezze prima dell'aggiornamento
        initial_train_len = len(iterative_train_dataset)
        initial_unlabel_len = len(unlabel_train_df)

        # Image index: indici delle immagini all'interno del train set
        train_df, unlabel_train_df = update_datasets(images_index, images_label, predictions_dataset, train_df, unlabel_train_df)

        # Lunghezze dopo l'aggiornamento
        final_train_len = len(iterative_train_dataset)
        final_unlabel_len = len(unlabel_train_df)

        with open('./ground_truth/my_train.csv', 'w+') as f1:
            train_df.to_csv(f1, header=False, index=False)

        f.write(f'Numero immagini aggiunte al train set: {final_train_len - initial_train_len}\n')
        f.write(f'Numero immagini rimosse dall\'unlabel set: {initial_unlabel_len - final_unlabel_len}\n')

        