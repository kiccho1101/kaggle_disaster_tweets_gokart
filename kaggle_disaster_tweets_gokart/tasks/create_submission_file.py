import glob
import datetime
import pandas as pd
import sklearn.ensemble
from kaggle_disaster_tweets_gokart import (
    Tweet,
    MakeEnsembleModel,
    MakeTrainSelectedFeatureData,
    MakeTestSelectedFeatureData,
)


class CreateSubmissionFile(Tweet):
    def requires(self):
        return dict(
            model=MakeEnsembleModel(),
            dftrain=MakeTrainSelectedFeatureData(),
            dftest=MakeTestSelectedFeatureData(),
        )

    def run(self):

        model: sklearn.ensemble.StackingClassifier = self.load("model")
        dftrain: pd.DataFrame = self.load("dftrain")
        dftest: pd.DataFrame = self.load("dftest")
        df: pd.DataFrame = pd.read_csv("./nlp-getting-started/sample_submission.csv")
        model.fit(dftrain.drop("target", axis=1).values, dftrain["target"])
        df["target"] = model.predict(dftest.values)

        submission_file_prefix: str = "./output/submission_{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d")
        )
        submission_no = len(glob.glob(submission_file_prefix + "_*.csv")) + 1
        submission_file_name = "{}_{}.csv".format(submission_file_prefix, submission_no)
        df.to_csv(submission_file_name, index=False)
        print("Sumission file: {} saved!".format(submission_file_name))
        self.dump(df)
