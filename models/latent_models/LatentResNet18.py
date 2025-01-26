import torch
import torchvision.models as models
from torch import nn
from tqdm import tqdm

class LatentResNet50:
    def __init__(self):
        # Carica il modello ResNet50 pre-addestrato
        self.model = models.resnet18(pretrained=True)
        
        # Costruisci l'estrattore di feature rimuovendo gli strati finali
        self.feature_extractor = nn.Sequential(
            *list(self.model.children())[:-2],  # Fino a 'layer4', senza il fully connected
            self.model.avgpool               # Aggiungi avgpool
        )
        
        # Definisci il dispositivo per l'esecuzione (CPU o GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sposta l'estrattore di feature sul dispositivo
        self.feature_extractor = self.feature_extractor.to(self.device)

    def extract_features(self, imageLoader):
        features_list = []        
        # Disabilita il calcolo del gradiente (inferenza)
        with torch.no_grad():
            for images, _ in tqdm(imageLoader):
                images = images.to(self.device)
                features = self.feature_extractor(images)
                
                # Aggiungi le feature estratte alla lista
                features_list.append(features)
        
        # Concatenare le feature estratte in un unico tensore
        features_tensor = torch.cat(features_list, dim=0)
        
        return features_tensor
