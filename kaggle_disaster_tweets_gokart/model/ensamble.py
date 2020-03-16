from logging import getLogger
from typing import Dict, List, Any, Tuple
import numpy as np
from lightgbm import LGBMClassifier

logging = getLogger(__name__)


class EnsambleModel:
    def __init__(
        self,
        model,
        weight: float,
        params: Dict[str, Any],
        fit_params: Dict[str, Any] = {},
    ):
        self.model = model
        self.weight: float = weight
        self.params: Dict[str, Any] = params
        self.fit_params: Dict[str, Any] = fit_params

    def fit(self, x: List[List[float]], y: List[List[float]]) -> None:
        self.model.fit(x, y, **self.fit_params)

    def predict(
        self, x: List[List[float]]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        return self.model.predict(x), self.model.predict_proba(x)


class Ensamble:
    def __init__(
        self, models: List[EnsambleModel],
    ):
        self.models: List[EnsambleModel] = models

    def fit(self, x: List[List[float]], y: List[List[float]]) -> None:
        for model in self.models:
            model.fit(x, y)

    def predict(
        self, x: List[List[float]]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        return (
            np.sum(
                [model.weight * np.array(model.predict(x)[0]) for model in self.models],
                axis=0,
            )
            / np.sum([model.weight for model in self.models]),
            np.sum(
                [model.weight * np.array(model.predict(x)[1]) for model in self.models],
                axis=0,
            )
            / np.sum([model.weight for model in self.models]),
        )
