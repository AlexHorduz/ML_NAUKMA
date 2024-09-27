from typing import Dict, Callable, Optional, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from models.base_model import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(
        self,
        metrics: Dict[str, Tuple[Callable, str]] = None,
        *args,
        **kwargs
    ):
        self.model = DecisionTreeClassifier(*args, **kwargs)
        self.metrics = metrics
        super().__init__(*args, **kwargs)
        
        self.history.update(
            {f"{subset}_{metric}": [] for subset in ["train", "val"] for metric in metrics.keys()}
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        self.model.fit(X_train, y_train)

        for metric_name, (metric_func, predict_type) in self.metrics.items():
            y_pred = getattr(self, predict_type)(X_train)

            assert y_train.shape == y_pred.shape, (
                f"train {predict_type}: {y_train.shape}, {y_pred.shape}"
            )
            metric_value = metric_func(y_train, y_pred)   
            self.history[f"train_{metric_name}"].append(metric_value)

            y_pred = getattr(self, predict_type)(X_val)
            assert y_val.shape == y_pred.shape, (
                f"val {predict_type}: {y_val.shape}, {y_pred.shape}"
            )
            metric_value = metric_func(y_val, y_pred)   
            self.history[f"val_{metric_name}"].append(metric_value)


    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        preds = np.expand_dims(preds, axis=1)
        return preds
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict_proba(X)
        preds = np.expand_dims(preds[:, 1], axis=1)
        return preds