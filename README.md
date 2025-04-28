# Project FoodX_251

## Description
This project focuses on the analysis and creation of complex models for image recognition using the FoodX_251 dataset. Throughout the project, various machine learning models were explored, performance on weak datasets was analyzed, and several aspects related to image quality were managed.

## Project Structure

- **`utils/`**: Contains the `requirements.txt` file for setting up the virtual environment.
- **`dataset/`**: Contains the necessary data for the project. Download and extract the following datasets into this folder:
  - `train_set`
  - `test_set`
  - `val_set_degraded`
- **`FoodX_251/`**: Includes data analysis, exploration of simple models, and the creation of the complex model.
- **`ground_truth/`**: Contains the ground truth files. Some of them have been modified to handle missing images and simplify data splitting processes.
- **`models/`**: Contains the model classes used and discarded throughout the project. The `trained_models/` subfolder includes the pre-trained models. Files ending with `-1` refer to noisy models.
- **`scripts/`**: Contains essential classes and scripts fundamental to the project.
- **`graphs/`**: Contains training graphs for the various models.

## Installation

1. **Create a virtual environment**:

   ```bash
   python -m venv environment_name
   ```

2. **Activate the virtual environment**:

   - On Windows:
     ```bash
     environment_name\Scripts\activate
     ```
   - On Linux/MacOS:
     ```bash
     source environment_name/bin/activate
     ```

3. **Install dependencies**:

   The `requirements.txt` file is located in the `utils` folder. To install all dependencies, run:

   ```bash
   pip install -r utils/requirements.txt
   ```

## Usage

1. **Prepare the data**: Download and extract the following datasets into the `dataset/` folder:
   - `train_set`
   - `test_set`
   - `val_set_degraded`

2. **Run data analysis**: The data analysis is included in the `FoodX_251` folder. This folder also contains the exploration of simple models and the creation of the complex model. Analysis related to degraded images can be found in the `degraded_images.ipynb` notebook.

3. **Model training**: Models are located in the `models/` folder. Pre-trained models are available in the `trained_models/` subfolder. You can also train models from scratch using the source code available in the `scripts/` folder.

4. **Training graphs**: The training graphs for various models are located in the `graphs/` folder.

5. **Updating Training Images and Cyclic Training of the Ensemble**

The script `ensamble_image_increment.py` manages the update of training images and cyclic training of the ensemble. This process consists of progressively adding new images to the training dataset, thus improving the model's accuracy over time.

#### Using the `ensamble_image_increment.py` Script

To update the training images and train the ensemble cyclically, simply run the script. The script will:

1. Add new images to the training dataset.
2. Retrain the ensemble models on the updated dataset.

Example of execution:

```bash
python ensamble_image_increment.py
```

## Using the `EnsambleModel` Class

The `EnsambleModel` class allows you to create an ensemble model, train individual models, and make predictions using the combined weight of each model.

### Initializing the Ensemble

To create an object of the `EnsambleModel` class, you must provide the following parameters:

- `models_name`: a list with the names of the models to be used in the ensemble.
- `pre_trained`: if set to `True`, loads the pre-trained weights for the models. If `False`, no pre-trained weights are loaded.
- `models_weights`: a list of weights for each model. The sum of the weights determines each model’s importance during prediction. This parameter is useful only if `pre_trained=True`.
- `num_classes`: the number of classes in the dataset (default: 251).

Example of initialization:

```python
ensemble = EnsambleModel(
    models_name=['resnet', 'efficientnet', 'vgg'],
    pre_trained=True,
    models_weights=[0.3, 0.4, 0.3],
    num_classes=251
)
```

### Training the Ensemble

To train the models in the ensemble, use the `train_ensamble()` method. The required parameters are:

- `train_dataset`: the training dataset (an object of the `ImageDataset` class).
- `lr`: learning rate.
- `num_epochs`: the number of training epochs (default: 10).
- `lc`: additional parameters for training management, if necessary.

Example of training:

```python
train_losses, val_losses, train_accuracies, val_accuracies = ensemble.train_ensamble(
    train_dataset=train_data,
    lr=0.001,
    num_epochs=10
)
```

This method trains the models specified in `models_name` and returns the losses and accuracies for both training and validation.

### Making Predictions with the Ensemble

To make predictions on the data, use the `predict()` method. Required parameters are:

- `image_dataset`: an object of the `ImageDataset` class containing the images to predict.
- `lc`: additional parameters for prediction management, if necessary.

Example of prediction:

```python
images_idx, images_label, predictions_confidences = ensemble.predict(image_dataset=test_data)
```

The method returns three lists:

- `images_idx`: the IDs of the images.
- `images_label`: the predicted labels.
- `predictions_confidences`: the probabilities associated with the predicted labels.

### Loading Pre-Trained Models

If `pre_trained=True`, the models are loaded from the pre-trained weights stored in the `./models/trained_models/` folder. The model name must match the name of the weight file (e.g., `resnet_-1.pth` for the ResNet model).

### Note on Noisy Models

Files ending with `-1` refer to models trained on noisy data. If you do not want to use noisy models, you can omit these files from the pre-trained weights.
