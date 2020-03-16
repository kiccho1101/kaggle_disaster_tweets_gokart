from logging import getLogger

import os
import datetime
import json
import pandas as pd
import numpy as np
import sklearn.metrics
from typing import List, Dict

from kaggle_disaster_tweets_gokart.base import Tweet
from kaggle_disaster_tweets_gokart.tasks.split_train_data import SplitTrainData
from kaggle_disaster_tweets_gokart.model.ensamble import Ensamble
from kaggle_disaster_tweets_gokart.model.make_ensamble_model import MakeEnsambleModel

logger = getLogger(__name__)


class CrossValidation(Tweet):
    def requires(self):
        return dict(split_data=SplitTrainData(), model=MakeEnsambleModel())

    def run(self):
        cv_result_path = "resource/cv_result.pkl"
        if os.path.exists(cv_result_path):
            result: pd.DataFrame = pd.read_pickle(cv_result_path)
        else:
            result: pd.DataFrame = pd.DataFrame()

        split_data: List[Dict[str, pd.DataFrame]] = self.load("split_data")

        scores: List[Dict[str, float]] = []
        for cv_num, d in enumerate(split_data):
            model: Ensamble = self.load("model")
            model.fit(
                d["train"].drop("target", axis=1).values, d["train"]["target"].values
            )
            y_pred, y_pred_proba = model.predict(
                d["eval"].drop("target", axis=1).values,
            )
            y_true = d["eval"]["target"].values

            score = {
                "f1_score": sklearn.metrics.f1_score(y_true, y_pred),
                "precision": sklearn.metrics.precision_score(y_true, y_pred),
                "recall": sklearn.metrics.recall_score(y_true, y_pred),
            }
            scores.append(score)
            print(f"================== CV No. {cv_num} =================")
            print(json.dumps(score, sort_keys=True, indent=4))

        result = pd.concat(
            [
                result,
                pd.DataFrame(
                    [
                        {
                            "time": datetime.datetime.now(),
                            "model": "+".join(
                                [
                                    "{}*{}".format(m.weight, str(m.model))
                                    for m in model.models
                                ]
                            ),
                            "avg_f1_score": np.mean(
                                [score["f1_score"] for score in scores]
                            ),
                            "avg_precision": np.mean(
                                [score["precision"] for score in scores]
                            ),
                            "avg_recall": np.mean(
                                [score["recall"] for score in scores]
                            ),
                            "scores": scores,
                            "features": list(split_data[0]["train"].columns),
                        }
                    ]
                ),
            ]
        )
        result.to_pickle(cv_result_path)
