import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scripts.ImageDataset import ImageDataset
from torch.utils.data import DataLoader

from utils.train_support import *
from utils.LogClass import LogClass


if __name__ == '__main__':
    lc = LogClass('log.txt')
    model_list = ['resnet18', 'efficientnet', 'vgg16']

    train_gt_path = './ground_truth/train_small.csv'
    train_unlabel_path = './ground_truth/train_unlabeled.csv'
    test_gt_path = './ground_truth/val_info.csv'

    train_image_path = './train_set/'
    test_image_path = './val_set/'

    iterative_train_df = pd.read_csv(train_gt_path, header=None, names=['image_id', 'label'])
    lc.write(f'Length of train dataset: {len(iterative_train_df)}')
    iterative_train_dataset = ImageDataset(iterative_train_df, train_image_path, train=True)

    train_unlabel_df = pd.read_csv(train_unlabel_path, header=None, names=['image_id', 'label'])
    lc.write(f'Length of unlabel dataset: {len(train_unlabel_df)}')
    train_unlabel_dataset = ImageDataset(train_unlabel_df, train_image_path, train=True,dataset_size=25000)
    lc.write('\n')
    for cycle in range(10):
        lc.write(f'Cycle: {cycle}')
        # Alleno i modelli, o se esistono li carico e restituisco le predizioni sul dataset unlabeled
        models_accuracies, models_predictions = train_models(model_list, iterative_train_dataset, train_unlabel_dataset, cycle, num_epochs=10, lc=lc)
        # Calcolo i pesi dei modelli in base all'accuratezza sul validation set
        weights = np.array(models_accuracies) / np.sum(models_accuracies)
        lc.write(f'Weights: ', data= {weights})

        # Calcolo l'agreement tra le predizioni dei modelli e il threshold
        images_idx, images_label = get_agreement_and_treshold(models_predictions, iterative_train_df, weights, confidence=0.75, lc=lc)

        # Aggiorno i dataset rimuovendo le immagini aggiunte al training set iterativo
        it_len_before = len(iterative_train_df)
        un_len_before = len(train_unlabel_df)
        iterative_train_dataset, train_unlabel_dataset = update_datasets(images_idx, images_label, train_unlabel_dataset, iterative_train_dataset)
        it_len_after = len(iterative_train_df)
        un_len_after = len(train_unlabel_df)
        lc.write(f'Number of images added to train set: {it_len_after - it_len_before}')
        lc.write(f'Number of images removed from unlabel set: {un_len_before - un_len_after}')
        lc.write('----------------------------------------------------------------------------------------------------------\n')
        # Save results
        iterative_train_df.to_csv('iterative_train.csv', index=False)
