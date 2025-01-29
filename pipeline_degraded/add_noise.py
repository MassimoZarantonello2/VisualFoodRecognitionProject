import os
import numpy as np
import io
from PIL import Image, ImageFilter
from torchvision import transforms
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
filename = "train_059371.jpg"
filename_out = "train_059371_2.jpg"
img_path = os.path.join(input_folder, filename)

img = Image.open(img_path).convert("RGB")
img_resized = transform(img)
img_resized = transforms.ToPILImage()(img_resized)

noise_functions = [add_noise, blur_image, add_jpeg_noise]

noisy_img = random.choice(noise_functions)(img_resized)
noisy_img.save(os.path.join(noisy_folder, filename_out))

original_images.append(np.array(img_resized).flatten())
noisy_images.append(np.array(noisy_img).flatten())

print(original_images)
print(noisy_images)
