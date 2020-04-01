from logging import getLogger
from lightgbm import LGBMClassifier
import pandas as pd
from kaggle_disaster_tweets_gokart import Tweet, MakeTrainFeatureData


logger = getLogger(__name__)


class MakeFeatureImportance(Tweet):
    def requires(self):
        return MakeTrainFeatureData()

    def run(self):
        df: pd.DataFrame = self.load()

        model = LGBMClassifier(
            boosting_type="gbdt",
            num_leaves=260,
            learning_rate=0.04424,
            feature_fraction=0.0871,
        )
        model.fit(df.drop("target", axis=1).values, df["target"])
        importance = pd.DataFrame(
            {
                "feature": df.drop("target", axis=1).columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        self.dump(importance)
