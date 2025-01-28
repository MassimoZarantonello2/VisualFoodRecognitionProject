import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=251):
        super(SimpleCNN, self).__init__()
        
        self.conv_block = nn.Sequential(
            # Primo blocco convoluzionale
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # input: 3 canali, output: 32 canali
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Secondo blocco convoluzionale
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # input: 32 canali, output: 64 canali
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Terzo blocco convoluzionale
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # input: 64 canali, output: 128 canali
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),  # Assumendo un ridimensionamento delle immagini a 28x28
            nn.ReLU(),
            nn.Linear(512, num_classes)  # 251 classi di output
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x