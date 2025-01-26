from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd

def print_errors(pred_labels, true_labels, images, normalized = False, side_images = 5):
    ltf = pd.read_csv('./ground_truth/foods_names.csv')
    ltf = ltf.drop(columns=['Index'])
    label_to_foods = ltf.to_dict()['Food']
    plt.figure(figsize=(10, 10))
    for i in range(side_images**2):
        true_label = label_to_foods[true_labels[i]]
        model_prediction = label_to_foods[pred_labels[i]]
        plt.subplot(side_images,side_images,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if true_label == model_prediction:
            plt.imshow(images[i])
            plt.xlabel(f'Pred: {model_prediction}, True: {true_label}', color='green')
        else:
            plt.imshow(images[i])
            plt.xlabel(f'Pred: {model_prediction}, True: {true_label}', color='red')

    plt.show()

