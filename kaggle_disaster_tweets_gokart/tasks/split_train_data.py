from logging import getLogger
from typing import Dict, List
import pandas as pd
import sklearn
from kaggle_disaster_tweets_gokart.base import Tweet
from kaggle_disaster_tweets_gokart.tasks.make_feature_data import MakeTrainFeatureData


logger = getLogger(__name__)


class SplitTrainData(Tweet):
    def requires(self):
        return MakeTrainFeatureData()

    def run(self):
        df: pd.DataFrame = self.load()
        kf = sklearn.model_selection.StratifiedKFold(
            n_splits=5, random_state=42, shuffle=True
        )
        split_data: List[Dict[str, pd.DataFrame]] = [
            {"train": df.iloc[train_index, :], "eval": df.iloc[eval_index, :]}
            for train_index, eval_index in kf.split(
                df.drop("target", axis=1), df["target"]
            )
        ]
        self.dump(split_data)
