import os
import pandas as pd
import numpy as np
import sys
sys.path.append('../')



image_list = os.listdir('./train_set')
label_images = pd.read_csv('./ground_truth/train_small.csv', header=None, names=['image_id', 'label'])

print('Total images:', len(image_list))
print('Total labeled images:', len(label_images))

# remove the labeled images from the list of all images
unlabeled_images = list(set(image_list) - set(label_images['image_id']))

print('Total unlabeled images:', len(unlabeled_images))

# save the list of unlabeled images as a csv file
df = pd.DataFrame(unlabeled_images)
df.columns = ['image_id']
df.to_csv('./ground_truth/unlabeled.csv', index=False)