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
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import matplotlib.pyplot as plt

from models.logistic_regression import LogisticRegressionModel
from models.KNN import KNNModel
from models.decision_tree import DecisionTreeModel

LABELS_MAPPING = {
    "human": 0,
    "animal": 1
}

RANDOM_SEED = 42

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


def vectorize_images(images_list: list[np.ndarray]):
    ''' Vectorizes images into a matrix of size (N, D), where N is the number of images, and D is the dimensionality of the image.'''
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    images = []
    for i in range(len(images_list)):
        image = images_list[i]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (224, 224))  # Resize to input size
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = preprocess_input(image)
        images.append(image)
    images = np.vstack(images)
    X = model.predict(images)
    return X


def validation_split(
        X: np.ndarray, y: np.ndarray, test_size: float, indices: List[int] = None, seed: int = 42,
) -> Tuple[np.ndarray]:
    ''' Splits data into train and test.'''
    if indices:
        return train_test_split(
            X, y, indices, test_size=test_size, random_state=seed, shuffle=True, stratify=y
        )
    else:
        return train_test_split(
            X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=y
        )


def create_model(model_name: str):
    ''' Creates a model of the specified name. 
    1. LogisticRegression
    2. KNN
    3. DecisionTree
    Args:
        model_name (str): Name of the model to use.
    Returns:
        model (object): Model of the specified name.
    '''
    if model_name == "LogisticRegression":
        model = LogisticRegressionModel(
            epochs=100,
            metrics={
                "bce_loss": (log_loss, "predict_proba"),
                "accuracy": (accuracy_score, "predict"),
                "roc_auc": (roc_auc_score, "predict_proba"),
            },
        )
    elif model_name == "KNN":
        model = KNNModel(
            metrics={
                "bce_loss": (log_loss, "predict_proba"),
                "accuracy": (accuracy_score, "predict"),
                "roc_auc": (roc_auc_score, "predict_proba"),
            },
            n_neighbours=4,
        )
    elif model_name == "DecisionTree":
        model = DecisionTreeModel(
            metrics={
                "bce_loss": (log_loss, "predict_proba"),
                "accuracy": (accuracy_score, "predict"),
                "roc_auc": (roc_auc_score, "predict_proba"),
            },
            max_depth=8,
            min_samples_leaf=5,
            min_samples_split=15,
        )
    else:
        raise ValueError(
            f"Model name {model_name} not recognised, use one of [LogisticRegression, KNN, DecisionTree]"
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
    model_name: str,
    validation_strategy: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scores_path: str = "scores/part2.json",
):
    os.makedirs("/".join(scores_path.split("/")[:-1]), exist_ok=True)

    confusion_matrix_save_folder = f"plots/part2/{model_name}/{validation_strategy}"

    if not os.path.exists(confusion_matrix_save_folder):
        os.makedirs(confusion_matrix_save_folder, exist_ok=True)

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
        model = create_model(model_name)
        X_train, X_val, y_train, y_val = validation_split(X_train, y_train, test_size=0.2, seed=RANDOM_SEED)

        model.train(X_train, y_train, X_val, y_val)
        model.visualize_training(f"plots/part2/{model_name}/{validation_strategy}")

        for metric_name, (metric_func, predict_type) in model.metrics.items():
            y_pred = getattr(model, predict_type)(X_test)
            metric_value = metric_func(y_test, y_pred)
            scores[model_name][validation_strategy][metric_name] = metric_value
        
        all_preds = model.predict(X_test)
        tn, fp, fn, tp = map(int, confusion_matrix(y_test, all_preds).ravel())

        save_confusion_matrix(tp=tp, fp=fp, tn=tn, fn=fn, filename=f"{confusion_matrix_save_folder}/confusion_matrix.png")
        scores[model_name][validation_strategy]["confusion_matrix"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    elif validation_strategy in ["k_fold", "stratified_k_fold"]:
        if validation_strategy == "k_fold":
            splitter = KFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
        else:
            splitter = StratifiedKFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
        models = []
        for fold, (train_index, val_index) in enumerate(splitter.split(X_train, y_train)):
            model = create_model(model_name)
            X_train_fold = X_train[train_index, :]
            y_train_fold = y_train[train_index, :]

            X_val_fold = X_train[val_index, :]
            y_val_fold = y_train[val_index, :]

            model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            model.visualize_training(f"plots/part2/{model_name}/{validation_strategy}/fold_{fold}")
            
            models.append(model)

        for metric_name, (metric_func, predict_type) in model.metrics.items():
            all_preds = []
            for model in models:
                y_pred = getattr(model, predict_type)(X_test)
                all_preds.append(y_pred)
            all_preds = np.array(all_preds).mean(axis=0)
            if predict_type == "predict":
                all_preds = all_preds.round().astype(int)

            metric_value = metric_func(y_test, all_preds)
            scores[model_name][validation_strategy][metric_name] = metric_value

        all_preds = []
        for model in models:
            y_pred = model.predict(X_test)
            all_preds.append(y_pred)
        all_preds = np.array(all_preds).mean(axis=0)
        all_preds = all_preds.round().astype(int)

        tn, fp, fn, tp = map(int, confusion_matrix(y_test, all_preds).ravel())

        save_confusion_matrix(tp=tp, fp=fp, tn=tn, fn=fn, filename=f"{confusion_matrix_save_folder}/confusion_matrix.png")
        scores[model_name][validation_strategy]["confusion_matrix"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    else:
        raise ValueError(
            f"Validation strategy {validation_strategy} not recognised, use one of [simple_split, k_fold, stratified_k_fold]"
        )

    with open(scores_path, "w") as f:
        json.dump(scores, f)

    return all_preds



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
    X_train, X_test, y_train, y_test, indices_train, indices_test = validation_split(X, y, test_size, list(range(X.shape[0])), RANDOM_SEED)

    test_images = [images[i] for i in indices_test]

    inverse_labels_mapping = {number: label for (label, number) in LABELS_MAPPING.items()}

    for validation_strategy in ["simple_split", "k_fold", "stratified_k_fold"]:
        preds = perform_training_and_evaluation(
            model_name=model_name,
            validation_strategy=validation_strategy,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        wrongly_classified_images_folder = f"plots/part2/{model_name}/{validation_strategy}/wrongly_classified"
        if not os.path.exists(wrongly_classified_images_folder):
            os.makedirs(wrongly_classified_images_folder, exist_ok=True)

        for ind in range(len(test_images)):
            if preds[ind] != y_test[ind]:
                pred_label = inverse_labels_mapping[preds[ind, 0]]
                true_label = inverse_labels_mapping[y_test[ind, 0]]

                plt.title(f"True label: {true_label}, Predicted: {pred_label}")
                plt.imshow(cv2.cvtColor(test_images[ind], cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.savefig(f"{wrongly_classified_images_folder}/image_{ind}.png")
                plt.close()

if __name__ == "__main__":
    main()