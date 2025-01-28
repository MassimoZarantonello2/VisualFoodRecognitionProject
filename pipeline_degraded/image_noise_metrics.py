import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sympy.stats import kurtosis

from image_enhancement import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import json

from pipeline_degraded.metric_utils import detect_noises
from scripts.ImageDataset import ImageDataset

load_dotenv()

test_path = '../ground_truth/new_val_info.csv'
test_image_path = os.getenv('TEST_IMAGE_PATH')

test_table = pd.read_csv(test_path, header=None, names=['image_id', 'label'])

test_dataset = ImageDataset(test_table, test_image_path, train=False)

results = []
laplacian_variances = []
gradient_means = []
gradient_stds = []
gdf_entropies = []
extreme_pixels_per = []
mad = []
kurt = []
variances = []
snr = []

for i in range(len(test_dataset)):
    image=test_dataset.get_image_by_index(i)
    blurry_metrics = detect_noises(image)

    laplacian_variances.append(blurry_metrics["laplacian_variance"])
    gradient_means.append(blurry_metrics["gradient_mean"])
    gradient_stds.append(blurry_metrics["gradient_std"])
    gdf_entropies.append(blurry_metrics["gdf_entropy"])
    extreme_pixels_per.append(blurry_metrics["extreme_pixel"])
    mad.append(blurry_metrics["mad"])
    kurt.append(blurry_metrics["kurt"])
    variances.append(blurry_metrics["variance"])
    snr.append(blurry_metrics["snr"])

    results.append({
        "image_id": test_table.iloc[i]['image_id'],
        "laplacian_variance": blurry_metrics["laplacian_variance"],
        "gradient_mean": blurry_metrics["gradient_mean"],
        "gradient_std": blurry_metrics["gradient_std"],
        "gdf_entropy": blurry_metrics["gdf_entropy"],
        "extreme_pixel": blurry_metrics["extreme_pixel"],
        "mad": blurry_metrics["mad"],
        "kurt": blurry_metrics["kurt"],
        "variance": blurry_metrics["variance"],
        "snr": blurry_metrics["snr"]
    })

    print(f"Image {i} processed.")

mean_laplacian_variance = np.mean(laplacian_variances)
mean_gradient_mean = np.mean(gradient_means)
mean_gradient_std = np.mean(gradient_stds)
mean_gdf_entropy = np.mean(gdf_entropies)
mean_extreme_pixels_per = np.mean(extreme_pixels_per)
mean_mad = np.mean(mad)
mean_kurt = np.mean(kurt)
mean_var = np.mean(variances)
mean_snr = np.mean(snr)

print("Mean Laplacian Variance:", mean_laplacian_variance)
print("Mean Gradient Mean:", mean_gradient_mean)
print("Mean Gradient Std:", mean_gradient_std)
print("Mean GDF Entropy:", mean_gdf_entropy)
print("Mean Extreme Pixels Percentage:", mean_extreme_pixels_per)
print("Mean Mad", mean_mad)
print("Mean Kurt", mean_kurt)
print("Mean Variance", mean_var)
print("Mean SNR", mean_snr)

output_path = "/Users/annamarika/Desktop/blurriness_metrics_degradato2.json"
with open(output_path, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"Results saved to {output_path}")