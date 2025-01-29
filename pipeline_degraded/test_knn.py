import numpy as np
import joblib
from PIL import Image
from torchvision import transforms
import random
import io

knn = joblib.load("knn_denoising_model.pkl")

transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor()
])

test_image_path = "/Users/annamarika/Desktop/val_set_degraded/val_000001.jpg"
img = Image.open(test_image_path)
img_resized = transform(img)
img_resized = transforms.ToPILImage()(img_resized)

noisy_array = np.array(img_resized ).flatten().reshape(1, -1)
denoised_array = knn.predict(noisy_array)

denoised_img = Image.fromarray(denoised_array.reshape(244, 244).astype(np.uint8))

img_resized.show(title="Original Image")
denoised_img.show(title="Denoised Image")