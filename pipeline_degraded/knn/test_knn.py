import numpy as np
import joblib
from PIL import Image
from torchvision import transforms
import random
import torch
import io

knn = joblib.load("knn_denoising_model.pkl")

transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor()
])

def denoise_image(image_path, knn_model):
    img = Image.open(image_path).convert("RGB")  # Assicura RGB
    img_tensor = transform(img)
    img_vector = img_tensor.numpy().reshape(1, -1)

    denoised_vector = knn_model.predict(img_vector)  # (1, H*W*3)

    denoised_image = denoised_vector.reshape(3, 244, 244)

    # Converti in immagine PIL
    denoised_image = np.clip(denoised_image, 0, 1)
    denoised_image_pil = transforms.ToPILImage()(torch.tensor(denoised_image))

    return denoised_image_pil


# Esempio di utilizzo
denoised_img = denoise_image("/Users/annamarika/Desktop/val_set_degraded/val_000001.jpg", knn)
denoised_img.show()  # Mostra l'immagine
