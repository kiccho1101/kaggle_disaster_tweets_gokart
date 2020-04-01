from logging import getLogger
import pandas as pd
from typing import Dict

from kaggle_disaster_tweets_gokart import Tweet, LoadTrainData, LoadTestData


logger = getLogger(__name__)


def preprocess(df):

    df["text_preprocessed"] = df["text"]

    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    replace_map: Dict[str, str] = {
        "<URL>": r"https?://\S+|www\.\S+",
        "<USER>": r"@\w+",
        "<SMILE>": rf"{eyes}{nose}[)]+|[)]+{nose}{eyes}",
        "<LOLFACE>": rf"{eyes}{nose}p+",
        "<SADFACE>": rf"{eyes}{nose}\(+|\)+{nose}{eyes}",
        "<NEUTRALFACE>": rf"{eyes}{nose}[\/|l*]",
        "<NUMBER>": rf"[-+]?[.\d]*[\d]+[:,.\d]*",
    }

    for k, v in replace_map.items():
        df["text_preprocessed"] = df["text_preprocessed"].str.replace(v, k)

    # Force splitting words
    df["text_preprocessed"] = df["text_preprocessed"].str.replace(
        "/", " / ", regex=False
    )

    # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
    df["text_preprocessed"] = df["text_preprocessed"].str.replace(
        r"([!?.]{2,})", lambda m: m.group(0)[:1] + " <REPEAT>"
    )

    return df


class PreprocessTrainData(Tweet):
    def requires(self):
        return dict(train=LoadTrainData())

    def run(self):
        dftrain: pd.DataFrame = self.load("train")
        dftrain = preprocess(dftrain)
        self.dump(dftrain)


class PreprocessTestData(Tweet):
    def requires(self):
        return dict(test=LoadTestData())

    def run(self):
        dftest: pd.DataFrame = self.load("test")
        dftest = preprocess(dftest)
        self.dump(dftest)
