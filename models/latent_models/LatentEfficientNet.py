import torch
import torchvision.models as models
from torch import nn
from tqdm import tqdm

class LatentEfficientNet:
    def __init__(self):
        # Carica il modello EfficientNet-B0 pre-addestrato
        self.model = models.efficientnet_b0(pretrained=True)
        
        # Costruisci l'estrattore di feature rimuovendo l'ultimo strato
        self.feature_extractor = nn.Sequential(
            self.model.features,             # Blocchi convoluzionali principali
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
                
                # Appiattisci le feature estratte per compatibilità (se necessario)
                features = features.view(features.size(0), -1)
                
                # Aggiungi le feature estratte alla lista
                features_list.append(features)
        
        # Concatenare le feature estratte in un unico tensore
        features_tensor = torch.cat(features_list, dim=0)
        
        return features_tensor
