import pandas as pd
from kaggle_disaster_tweets_gokart import Tweet, PreprocessTrainData, PreprocessTestData


class MakeOHEKeyword(Tweet):
    def requires(self):
        return dict(dftrain=PreprocessTrainData(), dftest=PreprocessTestData(),)

    def run(self):
        dftrain = self.load("dftrain")
        dftest = self.load("dftest")
        df = pd.get_dummies(
            pd.concat([dftrain, dftest], axis=0)["keyword"], prefix="fe_ohe_keyword"
        )
        df.columns = [col.replace("%", "") for col in df.columns]
        result = {
            "train": df.iloc[: dftrain.shape[0], :],
            "test": df.iloc[dftrain.shape[0] :, :],
        }

        self.dump(result)


class FETEKeywordTrain(Tweet):
    def requires(self):
        return dict(dftrain=PreprocessTrainData())

    def run(self):
        dftrain = self.load("dftrain")
        te_map = dftrain.groupby("keyword")["target"].mean()
        dftrain["fe_te_keyword"] = dftrain["keyword"].map(te_map).fillna(0)
        self.dump(dftrain[["fe_te_keyword"]])


class FETEKeywordTest(Tweet):
    def requires(self):
        return dict(dftrain=PreprocessTrainData(), dftest=PreprocessTestData())

    def run(self):
        dftrain = self.load("dftrain")
        dftest = self.load("dftest")
        te_map = dftrain.groupby("keyword")["target"].mean()
        dftest["fe_te_keyword"] = dftest["keyword"].map(te_map).fillna(0)
        self.dump(dftest[["fe_te_keyword"]])


class FEOHEKeywordTrain(Tweet):
    def requires(self):
        return MakeOHEKeyword()

    def run(self):
        self.dump(self.load()["train"])


class FEOHEKeywordTest(Tweet):
    def requires(self):
        return MakeOHEKeyword()

    def run(self):
        self.dump(self.load()["test"])
