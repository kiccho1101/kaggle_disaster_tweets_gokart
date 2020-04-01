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
    MakeTrainSelectedFeatureData,
    MakeTrainFeatureData,
    MakeEnsembleModel,
)

logger = getLogger(__name__)


class CrossValidation(Tweet):
    def requires(self):
        return dict(df=MakeTrainSelectedFeatureData(), model=MakeEnsembleModel())

    def run(self):
        df: pd.DataFrame = self.load("df")
        model: sklearn.ensemble.StackingClassifier = self.load("model")

        try:
            mlflow.end_run()
        except Exception:
            pass

        metrics = ["f1", "precision", "recall", "roc_auc"]

        with mlflow.start_run():
            mlflow.log_param("models", "+".join([m[0] for m in model.estimators]))
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_param("model", str(model))
            mlflow.log_param("features", str(list(df.drop("target", axis=1).columns)))
            result = sklearn.model_selection.cross_validate(
                model,
                df.drop("target", axis=1).values,
                df["target"].values,
                scoring=metrics,
                cv=5,
                verbose=1,
                n_jobs=-1,
            )

            mlflow.log_param("svg_fit_time", np.mean(result["fit_time"]))

            for metric in metrics:
                mlflow.log_metric(f"{metric}", np.mean(result[f"test_{metric}"]))
                for i, m in enumerate(result[f"test_{metric}"]):
                    mlflow.log_metric(f"{metric}_{i}", m)

        self.dump(result)
