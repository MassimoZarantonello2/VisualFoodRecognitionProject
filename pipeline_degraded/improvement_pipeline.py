import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from image_enhancement import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

from pipeline_degraded.metric_utils import detect_noises
from scripts.ImageDataset import ImageDataset

load_dotenv()

test_path = '../ground_truth/my_val_info.csv'
test_image_path = os.getenv('TEST_IMAGE_PATH')

test_table = pd.read_csv(test_path, header=None, names=['image_id', 'label'])

test_dataset = ImageDataset(test_table, test_image_path, train=False)

n = len(test_dataset)
for i in range(n):
    image = test_dataset.get_image_by_index(i)
    blurry_metrics = detect_noises(image)

    image = adaptive_gamma_correction(image)
    image = max_rgb(image)

    denoise_applied = False
    blurriness_applied = False

    if 200 <= blurry_metrics["laplacian_variance"] <= 5000:
        pass
    elif blurry_metrics["laplacian_variance"] > 5000:
        if not denoise_applied:
            image = denoise_salt_pepper(image)
            image = denoise_bilateral(image)
            denoise_applied = True
    elif blurry_metrics["laplacian_variance"] < 150:
        if not blurriness_applied:
            image = deblurring(image)
            blurriness_applied = True

    if 200 <= blurry_metrics["gradient_mean"] <= 1250:
        pass
    elif blurry_metrics["gradient_mean"] > 1250:
        if not denoise_applied:
            image = denoise_salt_pepper(image)
            image = denoise_bilateral(image)
            denoise_applied = True
    elif blurry_metrics["gradient_mean"] < 200:
        if not blurriness_applied:
            image = deblurring(image)
            blurriness_applied = True

    if blurry_metrics["gdf_entropy"] > 4.5:
        if not denoise_applied:
            image = denoise_salt_pepper(image)
            image = denoise_bilateral(image)
            denoise_applied = True

    if blurry_metrics["gradient_std"] < 450:
        if not blurriness_applied:
            image = deblurring(image)
            blurriness_applied = True
    print(f"Image {i} processed.")
    image.save(f"/Users/annamarika/Desktop/improvement_degradated/{test_table.iloc[i]['image_id']}")
