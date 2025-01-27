from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class ImageDataset(Dataset):
    '''
    Classe per la creazione di un dataset personalizzato.
    metodi:
        __init__: Costruttore della classe.
        __len__: Restituisce la lunghezza del dataset.
        __getitem__: Restituisce un'immagine e la sua etichetta.
    '''

    def __init__(self, gt_dataframe, image_path, train=True, dataset_size=None):
        '''
        if labels = True: il dataset contiene le label
        if train = True: il dataset è di train e può essere diviso in train e validation 
        '''
        self.dataframe = gt_dataframe
        self.image_path = image_path
        if dataset_size:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
            self.dataframe = self.dataframe[:dataset_size]
        # Trasformazioni
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((244, 244)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomCrop(224, padding=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((244, 244)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_id = self.dataframe.iloc[idx, 0]
        image_path = os.path.join(self.image_path, image_id)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        if self.dataframe.shape[1] > 1:
            label = self.dataframe.iloc[idx, 1]
            return image, label
        else:
            return image
        
    def get_image_by_index(self, idx):
        image_id = self.dataframe.iloc[idx, 0]
        image_path = os.path.join(self.image_path, image_id)
        image = Image.open(image_path)
        return image
    
    def get_image_by_id(self, image_id):
        image_path = os.path.join(self.image_path, image_id)
        image = Image.open(image_path)
        return image
    
    def get_image_id(self, idx):
        return self.dataframe.iloc[idx, 0]
    
    def add_image(self, image_id, label=None):
        if label is None:
            label = -1
        new_row = pd.DataFrame({'image_id': [image_id], 'label': [label]})
        self.dataframe = pd.concat([self.dataframe, new_row], ignore_index=True)
        return self.dataframe
    
    def remove_image(self, image_id):
        self.dataframe = self.dataframe[self.dataframe['image_id'] != image_id]
        return self.dataframe
    
    def get_all_labels(self):
        return self.dataframe['label'].tolist()
    
    def random_sample(self, size):
        return ImageDataset(self.dataframe.sample(n=size).reset_index(drop=True), self.image_path, train=True)

if __name__ == '__main__':
    train_gt_path = './ground_truth/train_small.csv'
    test_gt_path = './ground_truth/new_val_info.csv'
    train_unlabel_path = './ground_truth/unlabeled.csv'

    train_image_path = './train_set/'
    test_image_path = './val_set/'

    train_gt_df = pd.read_csv(train_gt_path, header=None, names=['image_id', 'label'])
    test_gt_df = pd.read_csv(test_gt_path, header=None, names=['image_id', 'label'])
    train_dataset = ImageDataset(train_gt_df, train_image_path, train=True, labels = True)
    test_dataset = ImageDataset(test_gt_df, test_image_path, train=False, labels = True)

    print('---- Dataset summary ----')
    print('Train dataset size:', len(train_dataset))
    print('Test dataset size:', len(test_dataset))

    def update_datasets(images_idx, images_label, predictions_dataset, train_dataset, unlabel_df):
        for i in range(len(images_idx)):
            # Ottengo l'immagine e la sua label dal dataset delle predizioni
            image_idx = images_idx[i]
            image_label = images_label[i]
            image_id = predictions_dataset.get_image_id(image_idx)
            # Aggiungo l'immagine al dataset di train iterativo
            train_dataset.add_image(image_id, image_label)
            # Rimuovo l'immagine dal dataset delle immagini senza label così da non aggiungerla nuovamente
            unlabel_df = unlabel_df[unlabel_df['image_id'] != image_id]

        return train_dataset, unlabel_df
    
    # Test per la funzione update_datasets
    predictions_dataset = test_dataset
    train_dataset = train_dataset
    unlabel_df = pd.read_csv(train_unlabel_path, header=None, names=['image_id'])
    images_idx = [0, 1, 2]
    images_label = [0, 1, 2]
    train_dataset, unlabel_df = update_datasets(images_idx, images_label, predictions_dataset, train_dataset, unlabel_df)
    print('---- Dataset summary ----')
    print('Train dataset size:', len(train_dataset))
    print('Unlabel dataset size:', len(unlabel_df))
    print('Test dataset size:', len(test_dataset))

