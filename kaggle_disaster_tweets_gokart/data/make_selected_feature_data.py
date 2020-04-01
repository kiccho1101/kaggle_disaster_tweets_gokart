import luigi
from kaggle_disaster_tweets_gokart import (
    Tweet,
    MakeTrainFeatureData,
    MakeTestFeatureData,
    MakeFeatureImportance,
)


class MakeTrainSelectedFeatureData(Tweet):
    min_importance = luigi.IntParameter(default=0)
    additional_cols = luigi.ListParameter()

    def requires(self):
        return dict(df=MakeTrainFeatureData(), importance=MakeFeatureImportance())

    def run(self):
        df = self.load("df")
        importance = self.load("importance")
        selected_cols = ["target"] + importance.query(
            f"importance > {self.min_importance}"
        )["feature"].sort_values().values.tolist()
        for col in self.additional_cols:
            if col not in selected_cols:
                selected_cols.append(col)
        with open("./resource/INPUT_LEN.txt", "w") as f:
            f.write("%d" % (len(selected_cols) - 1))
        self.dump(df[selected_cols])


class MakeTestSelectedFeatureData(Tweet):
    min_importance = luigi.IntParameter(default=0)
    additional_cols = luigi.ListParameter()

    def requires(self):
        return dict(df=MakeTestFeatureData(), importance=MakeFeatureImportance())

    def run(self):
        df = self.load("df")
        importance = self.load("importance")
        selected_cols = (
            importance.query(f"importance > {self.min_importance}")["feature"]
            .sort_values()
            .values.tolist()
        )
        for col in self.additional_cols:
            if col not in selected_cols:
                selected_cols.append(col)
        self.dump(df[selected_cols])
