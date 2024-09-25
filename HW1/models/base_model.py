from typing import Optional

import numpy as np

class BaseModel:
    def __init__(self, *args, **kwargs):
        self.history = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def visualize_training(self, save_folder: str) -> None:
        pass