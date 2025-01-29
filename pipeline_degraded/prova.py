import os
import io
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from sklearn.neighbors import KNeighborsRegressor
import joblib
import random


transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor()
])

def add_noise(im):
    img_array = np.array(im, dtype=np.float32)
    noise = np.random.normal(0, 60, img_array.shape)
    noisy_image_array = img_array + noise
    noisy_image_array = np.clip(noisy_image_array, 0, 255)
    noisy_image = Image.fromarray(np.uint8(noisy_image_array))
    return noisy_image


def blur_image(image):
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=4))
    return blurred_image


def add_jpeg_noise(image, quality=8):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    noisy_image = Image.open(buffer)
    return noisy_image


input_folder = "/Users/annamarika/Desktop/small_set"
noisy_folder = "/Users/annamarika/Desktop/small_noise_set"
os.makedirs(noisy_folder, exist_ok=True)

original_images = []
noisy_images = []

noise_functions = [add_noise, blur_image, add_jpeg_noise]

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        img_resized = transform(img)
        img_resized = transforms.ToPILImage()(img_resized)

        noisy_img = random.choice(noise_functions)(img_resized)
        noisy_img_tensor = transform(noisy_img)
        noisy_img.save(os.path.join(noisy_folder, filename))

        original_images.append(np.array(img_resized).flatten())
        noisy_images.append(np.array(noisy_img).flatten())

print("End of for")
original_images = np.array(original_images)
noisy_images = np.array(noisy_images)

np.save("original_images.npy", original_images)
np.save("noisy_images.npy", noisy_images)

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(noisy_images, original_images)

joblib.dump(knn, "knn_denoising_model.pkl")
print("Modello addestrato")
"""
original_images = np.load("original_images.npy")
noisy_images = np.load("noisy_images.npy")

joblib.dump(knn, "knn_denoising_model.pkl")
knn_loaded = joblib.load("knn_denoising_model.pkl")

"""


