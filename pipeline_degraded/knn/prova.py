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


input_folder = "/Users/annamarika/Desktop/train_set"
# noisy_folder = "/Users/annamarika/Desktop/small_noise_set_2"
# os.makedirs(noisy_folder, exist_ok=True)

original_images = []
noisy_images = []
i = 0
noise_functions = [add_noise, blur_image, add_jpeg_noise]
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")

        img_tensor = transform(img)
        img_resized = transforms.ToPILImage()(img_tensor)

        noisy_img = random.choice(noise_functions)(img_resized)
        noisy_img_tensor = transform(noisy_img)

        # noisy_img.save(os.path.join(noisy_folder, filename))

        original_images.append(img_tensor.numpy())
        noisy_images.append(noisy_img_tensor.numpy())
        print(f"fatto immagine {i}")
        i += 1

print("End of for")

# sono ndarray con dim `num_immagini` x 3 x 244 x 244
original_images = np.array(original_images)
noisy_images = np.array(noisy_images)

np.save("original_images2.npy", original_images)
np.save("noisy_images2.npy", noisy_images)

N, C, H, W = original_images.shape

original_images = original_images.reshape(N, -1)
# in questo modo original_images diventa N x 178608(ovvero 3x244x244)
noisy_images = noisy_images.reshape(N, -1)

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(noisy_images, original_images)

joblib.dump(knn, "knn_denoising_model2.pkl")

print("Modello addestrato")


"""
Lascio qui codice per aprire modelli salvati:

original_images = np.load("original_images.npy")
noisy_images = np.load("noisy_images.npy")

joblib.dump(knn, "knn_denoising_model.pkl")
knn_loaded = joblib.load("knn_denoising_model.pkl")

"""


