import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from image_enhancement import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import json

from scripts.ImageDataset import ImageDataset

load_dotenv()

test_path = '../ground_truth/new_val_info.csv'
test_image_path = os.getenv('TEST_IMAGE_PATH')
# test_image_path = os.getenv('TRAIN_SET')

test_table = pd.read_csv(test_path, header=None, names=['image_id', 'label'])

test_dataset = ImageDataset(test_table, test_image_path, train=False)

exemple = test_dataset.get_image_by_id("val_000001.jpg")
