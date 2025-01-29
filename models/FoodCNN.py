import torch
from torch import nn, optim
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision import models
from torch.utils.data import DataLoader
from scripts.ImageDataset import ImageDataset
from models.SimpleCNN import SimpleCNN
from sklearn.model_selection import train_test_split
from pipeline_degraded.image_improvement import image_improvement
import os

class FoodCNN():
    def __init__(self, model_name, num_classes=251):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.model = self._initialize_model()
        self.criterion = nn.CrossEntropyLoss()

    def _initialize_model(self):
        """Inizializza il modello in base al nome specificato."""
        if self.model_name == 'resnet18':
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
            for param in model.parameters():
                param.requires_grad = True
            self.parameters = model.fc.parameters()
        elif self.model_name == 'efficientnet':
            model = models.efficientnet_b0(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_features, self.num_classes)
            for param in model.classifier[-1].parameters():
                param.requires_grad = True
            self.parameters = model.classifier.parameters()
        elif self.model_name == 'vgg16':
            weights = VGG16_Weights.IMAGENET1K_V1
            model = vgg16(weights=weights)
            for param in model.parameters():
                param.requires_grad = False
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_features, self.num_classes)
            for param in model.classifier[-1].parameters():
                param.requires_grad = True
            self.parameters = model.classifier.parameters()
        elif self.model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
            for param in model.fc.parameters():
                param.requires_grad = True
            self.parameters = model.fc.parameters()
        elif self.model_name == 'simplecnn':
            model = SimpleCNN(self.num_classes)
            self.parameters = model.parameters()
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        return model.to(self.device)

    def evaluate(self, val_loader):
        '''
        Funzione per la valutazione del modello. Prende in input il dataloader di validazione e il criterio di loss.
        Restituisce la loss e l'accuratezza.
        '''
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        return val_loss / len(val_loader), val_accuracy

    def train_model(self, train_dataset, validation=0, num_epochs=10, lr=0.001, save_path='./models/trained_models/', cycle = 0):
        '''
        Funzione per il training del modello. Prende in input il dataloader di training e di validazione, il numero di epoche
        e il learning rate. Restituisce le liste di loss di training e validazione.
        '''
        img_path = train_dataset.image_path
        if validation > 0:
            train_dataset, val_dataset = train_test_split(train_dataset.dataframe, test_size=validation, random_state=42)
            train_dataset = ImageDataset(train_dataset,image_path=img_path, train=True)
            val_dataset = ImageDataset(val_dataset,image_path=img_path, train=False)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters, lr=lr)

        # scheduler per rendere il learning rate dinamico (decrescente)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        total_step = len(train_loader)
        train_loss = []
        val_loss = []
        train_accuracies = []
        val_accuracies = []

        # Barra di progresso per ogni epoca
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Barra di progresso per ogni epoca
            epoch_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', total=total_step, ncols=100)

            for i, (images, labels) in enumerate(epoch_progress_bar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calcolare l'accuratezza del training
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()

                # Aggiornamento della barra di progresso per il batch
                epoch_progress_bar.set_postfix(loss=loss.item(), accuracy=(correct / total), refresh=True)

            train_accuracy = correct / total
            print(f"Train Accuracy: {train_accuracy:.4f}")
            train_loss.append(running_loss / len(train_loader))
            train_accuracies.append(train_accuracy)

            # Valutazione
            val_loss_epoch, val_accuracy = self.evaluate(val_loader)
            print(f"Validation Loss: {val_loss_epoch:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            val_loss.append(val_loss_epoch)
            val_accuracies.append(val_accuracy)

            scheduler.step()

            if os.path.exists(save_path + f'{self.model_name}_{cycle-1}.pth'):
                os.remove(save_path + f'{self.model_name}_{cycle-1}.pth')                          
            torch.save(self.model.state_dict(), save_path + f'{self.model_name}_{cycle}.pth')
        return train_loss, val_loss, train_accuracies, val_accuracies
    
    def predict(self, image_dataset):
        '''
        Funzione per la predizione del modello. Prende in input il dataloader di test. Restituisce le predizioni.
        '''
        images_loader = DataLoader(image_dataset, batch_size=32)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in tqdm(images_loader):
                if isinstance(batch, (tuple, list)):
                # Estrai solo le immagini
                    images = batch[0]
                else:
                # Il batch è composto solo da immagini
                    images = batch

                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs.data, 1)
                predictions.extend(probs.cpu().numpy())
        return predictions
    
    def return_model_accuracy(self, test_loader):
        '''
        Funzione per il calcolo dell'accuratezza del modello. Prende in input il dataloader di test.
        '''
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
    
    def load_model(self, path):
        '''
        Funzione per il caricamento di un modello salvato. Prende in input il path del modello.
        '''
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        return self.model