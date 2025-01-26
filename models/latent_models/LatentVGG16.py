import torch
import torchvision.models as models
from torch import nn

class LatentVGG16:
    def __init__(self):
        # Carica il modello VGG16 pre-addestrato
        self.model = models.vgg16(pretrained=True)
        
        # Costruisci l'estrattore di feature rimuovendo gli strati finali
        self.feature_extractor = nn.Sequential(
            *list(self.model.children())[:-2],  # Fino a 'features', senza il fully connected
            self.model.avgpool               # Aggiungi avgpool
        )
        
        # Definisci il dispositivo per l'esecuzione (CPU o GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sposta l'estrattore di feature sul dispositivo
        self.feature_extractor = self.feature_extractor.to(self.device)

    def extract_features(self, imageLoader):
        print(f'{imageLoader.dataset.__len__()} images to process')
        features_list = []
        
        # Disabilita il calcolo del gradiente (inferenza)
        with torch.no_grad():
            for images, _ in imageLoader:
                images = images.to(self.device)
                features = self.feature_extractor(images)
                
                # Aggiungi le feature estratte alla lista
                features_list.append(features)
        
        # Concatenare le feature estratte in un unico tensore
        features_tensor = torch.cat(features_list, dim=0)
        
        return features_tensor