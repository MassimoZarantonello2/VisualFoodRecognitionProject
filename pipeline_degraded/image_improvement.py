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

def image_improvement(image):
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
    if not blurriness_applied and not denoise_applied:
        image = denoise_salt_pepper(image)
        image = denoise_bilateral(image)
        image = blurriness(image)
    return image