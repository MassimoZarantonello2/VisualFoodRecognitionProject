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
from image_improvement import image_improvement

load_dotenv()

test_path = '../ground_truth/my_val_info.csv'
test_image_path = os.getenv('TEST_IMAGE_PATH')

test_table = pd.read_csv(test_path, header=None, names=['image_id', 'label'])

test_dataset = ImageDataset(test_table, test_image_path, train=False)

n = len(test_dataset)
for i in range(n):
    image = test_dataset.get_image_by_index(i)
    image_hence = image_improvement(image)
    print(f"Image {i} processed.")
    image_hence.save(f"/Users/annamarika/Desktop/improvement_degradated/{test_table.iloc[i]['image_id']}")
