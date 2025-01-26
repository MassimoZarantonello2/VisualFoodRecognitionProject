import pandas as pd
import os

val_im_gt = pd.read_csv('ground_truth/new_val_info.csv', header=None)
val_im_gt.columns = ['image', 'label']

val_im = os.listdir('val_set')

val_im_gt_list = val_im_gt['image'].tolist()
non_present_images = []
for i in val_im_gt_list:
    if i not in val_im:
        non_present_images.append(i)

print('Number of images in the ground truth that are not in the validation set folder:', len(non_present_images))
print('Images:', non_present_images)
val_im_gt = val_im_gt[~val_im_gt['image'].isin(non_present_images)]
val_im_gt.to_csv('ground_truth/new_val_info.csv', header=False, index=False)