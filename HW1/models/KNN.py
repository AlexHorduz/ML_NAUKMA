from typing import Dict, Callable, Optional, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from models.base_model import BaseModel

class KNNModel(BaseModel):
    def __init__(
        self,
        metrics: Dict[str, Tuple[Callable, str]] = None,
        n_neighbours: int = 4,
        *args,
        **kwargs
    ):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbours, *args, **kwargs)
        self.metrics = metrics
        super().__init__(*args, **kwargs)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        self.model.fit(X_train, y_train)


    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        preds = np.expand_dims(preds, axis=1)
        return preds
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict_proba(X)
        preds = np.expand_dims(preds[:, 1], axis=1)
        return preds