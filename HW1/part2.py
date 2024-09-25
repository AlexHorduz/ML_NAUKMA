from typing import Tuple, List
import os
import json

import click
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix
import pandas as pd
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from models.base_model import BaseModel
from models.logistic_regression import LogisticRegressionModel

LABELS_MAPPING = {
    "human": 0,
    "animal": 1
}

def load_data(image_folder: str, label_file: str) -> Tuple[List[np.ndarray], np.ndarray]:
    ''' Loads images and labels from the specified folder and file.'''
    # load labels file
    metadata = pd.read_csv(label_file, sep="|")
    labels = metadata["label"].map(LABELS_MAPPING).values

    # load corresponding images
    images = []
    
    for image_path in metadata["image_name"].values:
        image = cv2.imread(f"{image_folder}/{image_path}") # HxWxC in BGR format
        images.append(image)

    return images, labels


def vectorize_images(images: np.ndarray):
    ''' Vectorizes images into a matrix of size (N, D), where N is the number of images, and D is the dimensionality of the image.'''
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    for i in range(len(images)):
        image = images[i]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (224, 224))  # Resize to input size
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = preprocess_input(image)
        images[i] = image
    images = np.vstack(images)
    X = model.predict(images)
    return X


def validation_split(X: np.ndarray, y: np.ndarray, test_size: float, seed: int = 42):
    ''' Splits data into train and test.'''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_model(model_name: str):
    ''' Creates a model of the specified name. 
    1. Use your LinearRegression implementation,
    2. TODO
    3. 
    Args:
        model_name (str): Name of the model to use.
    Returns:
        model (object): Model of the specified name.
    '''
    if model_name == "LogRegr":
        model = LogisticRegressionModel(
            epochs=100,
            metrics={
                "bce_loss": (log_loss, "predict_proba"),
                "accuracy": (accuracy_score, "predict"),
                "roc_auc": (roc_auc_score, "predict_proba"),
            },
        )
    elif model_name == "KNN":
        model = None
    elif model_name == "DecisionTree":
        model = None
    else:
        raise ValueError(
            f"Model name {model_name} not recognised, use one of [LogRegr, KNN, DecisionTree]"
        )
    return model

def save_confusion_matrix(tp, fp, tn, fn, filename="confusion_matrix.png"):
    confusion_matrix = np.array([[tp, fn], [fp, tn]])
    

    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap="Blues")

    plt.colorbar(cax)


    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')


    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Positive', 'Negative'])
    ax.set_yticklabels(['Positive', 'Negative'])

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='black', fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, format="png", dpi=300)
    plt.close()

def perform_training_and_evaluation(
    model: BaseModel,
    validation_strategy: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scores_path: str = "scores/part2.json",
):
    os.makedirs("/".join(scores_path.split("/")[:-1]), exist_ok=True)
    
    model_name = model.__class__.__name__

    if os.path.exists(scores_path):
        try:
            with open(scores_path, "r") as f:
                scores = json.load(f)
        except json.decoder.JSONDecodeError as e:
            scores = {}
    else:
        scores = {}
    
    scores[model_name] = scores.get(model_name, dict())
    scores[model_name][validation_strategy] = dict()

    if validation_strategy == "simple_split":
        X_train, X_val, y_train, y_val = validation_split(X_train, y_train, test_size=0.2, seed=42)

        model.train(X_train, y_train, X_val, y_val)
        model.visualize_training(f"plots/part2/{model_name}/simple_split")

        for metric_name, (metric_func, predict_type) in model.metrics.items():
            y_pred = getattr(model, predict_type)(X_test)
            metric_value = metric_func(y_test, y_pred)
            scores[model_name][validation_strategy][metric_name] = metric_value
        y_pred = model.predict(X_test)

        tn, fp, fn, tp = map(int, confusion_matrix(y_test, y_pred).ravel())
        save_confusion_matrix(tp=tp, fp=fp, tn=tn, fn=fn, filename=f"plots/part2/{model_name}/simple_split/confusion_matrix.png")
        scores[model_name][validation_strategy]["confusion_matrix"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    elif validation_strategy == "k_fold":
        pass
    elif validation_strategy == "stratified_k_fold":
        pass
    else:
        raise ValueError(
            f"Validation strategy {validation_strategy} not recognised, use one of [simple_split, k_fold, stratified_k_fold]"
        )

    with open(scores_path, "w") as f:
        json.dump(scores, f)



        




@click.command()
@click.option("--image_folder", type=str, help="Path to the folder containing images")
@click.option("--label_file", type=str, help="Path to the file containing labels")
@click.option("--model_name", type=str, help="Name of the model to use")
@click.option("--test_size", type=float, default=0.2, help="Size of the test split")
def main(image_folder: str, label_file: str, model_name: str, test_size: float):

    # Create dataset of image <-> label pairs
    images, labels = load_data(image_folder, label_file)


    # preprocess images and labels
    X = vectorize_images(images)

    y = np.expand_dims(labels, 1)

    # split data into train and test
    X_train, X_test, y_train, y_test = validation_split(X, y, test_size)

    # create model
    model = create_model(model_name)

    for validation_strategy in ["simple_split"]:
        perform_training_and_evaluation(
            model=model,
            validation_strategy=validation_strategy,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )


    # Train model using different validation strategies (refere to https://scikit-learn.org/stable/modules/cross_validation.html)
    # 1. Train, validation, test splits: so you need to split train into train and validation 
    # 2. K-fold cross-validation: apply K-fold cross-validation on train data
    # 3. Leave-one-out cross-validation: apply Leave-one-out cross-validation on train data



    # Make error analysis
    # 1. Plot the first 10 test images, and on each image plot the corresponding prediction
    # 2. Plot the confusion matrix



if __name__ == "__main__":
    main()