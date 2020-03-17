import luigi
from kaggle_disaster_tweets_gokart import (
    Tweet,
    MakeTrainFeatureData,
    MakeTestFeatureData,
    MakeFeatureImportance,
)


class MakeTrainSelectedFeatureData(Tweet):
    min_importance = luigi.IntParameter(default=0)

    def requires(self):
        return dict(df=MakeTrainFeatureData(), importance=MakeFeatureImportance())

    def run(self):
        df = self.load("df")
        importance = self.load("importance")
        selected_df = df[
            ["target"]
            + importance.query(f"importance > {self.min_importance}")["feature"]
            .sort_values()
            .values.tolist()
        ]
        self.dump(selected_df)


class MakeTestSelectedFeatureData(Tweet):
    min_importance = luigi.IntParameter(default=0)

    def requires(self):
        return dict(df=MakeTestFeatureData(), importance=MakeFeatureImportance())

    def run(self):
        df = self.load("df")
        importance = self.load("importance")
        selected_df = df[
            importance.query(f"importance > {self.min_importance}")["feature"]
            .sort_values()
            .values.tolist()
        ]
        self.dump(selected_df)
