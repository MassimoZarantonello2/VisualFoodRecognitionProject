import os
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from delete_noise import is_image_noisy


from pipeline_degraded.metric_utils import detect_noises
from scripts.ImageDataset import ImageDataset

load_dotenv()

test_path = '/Users/annamarika/PycharmProjects/VisualFoodRecognitionProject/ground_truth/my_val_info.csv'
test_image_path = os.getenv('TEST_IMAGE_PATH')
save_dir = "/Users/annamarika/Desktop/new_val/"

test_table = pd.read_csv(test_path, header=None, names=['image_id', 'label'])
test_dataset = ImageDataset(test_table, test_image_path, train=False)

os.makedirs(save_dir, exist_ok=True)
filtered_images = []

n = len(test_dataset)
for i in range(n):
    image = test_dataset.get_image_by_index(i)

    is_noisy = is_image_noisy(image)


    print(f"Image {i} processed.")
    if is_noisy:
        continue

    image_path = os.path.join(save_dir, test_table.iloc[i]['image_id'])
    image.save(image_path)
    filtered_images.append(test_table.iloc[i])

filtered_df = pd.DataFrame(filtered_images)
filtered_csv_path = os.path.join(save_dir, "filtered_images.csv")
filtered_df.to_csv(filtered_csv_path, index=False, header=False)

print(f"Salvate {len(filtered_images)} immagini e il file CSV in {filtered_csv_path}")