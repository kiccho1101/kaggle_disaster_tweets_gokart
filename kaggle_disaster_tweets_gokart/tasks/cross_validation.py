from logging import getLogger

import os
import datetime
import json
import pandas as pd
import numpy as np
import sklearn.metrics
from typing import List, Dict
import luigi
import mlflow
import mlflow.sklearn

import sklearn.ensemble
from kaggle_disaster_tweets_gokart import (
    Tweet,
    SplitTrainData,
    MakeEnsembleModel,
)

logger = getLogger(__name__)


class CrossValidation(Tweet):
    def requires(self):
        return dict(split_data=SplitTrainData(), model=MakeEnsembleModel())

    def run(self):

        models: List = []
        split_data: List[Dict[str, pd.DataFrame]] = self.load("split_data")
        scores: List[Dict[str, float]] = []

        try:
            mlflow.end_run()
        except Exception:
            pass

        with mlflow.start_run():
            model: sklearn.ensemble.StackingClassifier = self.load("model")
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_param("model", str(model))
            mlflow.log_param("models", "+".join([m[0] for m in model.estimators]))
            mlflow.log_param("features", str(list(split_data[0]["train"].columns)))
            for cv_num, d in enumerate(split_data):
                model: sklearn.ensemble.StackingClassifier = self.load("model")
                model.fit(
                    d["train"].drop("target", axis=1).values,
                    d["train"]["target"].values,
                )
                y_pred = model.predict(d["eval"].drop("target", axis=1).values,)
                y_true = d["eval"]["target"].values

                models.append(model)
                score = {
                    "f1_score": sklearn.metrics.f1_score(y_true, y_pred),
                    "precision": sklearn.metrics.precision_score(y_true, y_pred),
                    "recall": sklearn.metrics.recall_score(y_true, y_pred),
                }
                mlflow.log_metrics(
                    {
                        f"f1_score_{cv_num}": sklearn.metrics.f1_score(y_true, y_pred),
                        f"precision_{cv_num}": sklearn.metrics.precision_score(
                            y_true, y_pred
                        ),
                        f"recall_{cv_num}": sklearn.metrics.recall_score(
                            y_true, y_pred
                        ),
                    }
                )
                scores.append(score)
                print(f"================== CV No. {cv_num} =================")
                print(json.dumps(score, sort_keys=True, indent=4))
                print(f"=============================================")

            mlflow.log_metrics(
                {
                    "f1_score": np.mean([score["f1_score"] for score in scores]),
                    "precision": np.mean([score["precision"] for score in scores]),
                    "recall": np.mean([score["recall"] for score in scores]),
                }
            )
        result = {"models": models, "scores": scores}
        self.dump(result)
