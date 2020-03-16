from logging import getLogger

import pandas as pd

import re
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from kaggle_disaster_tweets_gokart.base import Tweet
from kaggle_disaster_tweets_gokart.tasks.preprocess_data import (
    PreprocessTrainData,
    PreprocessTestData,
)

logger = getLogger(__name__)


def fe_basic(df: pd.DataFrame):
    prefix: str = "fe_basic_"
    df[prefix + "length"] = df["text"].map(lambda x: len(x))
    df[prefix + "url_num"] = df["text_preprocessed"].map(
        lambda x: len(re.findall(r"\<URL\>", x))
    )

    nltk.download("stopwords")
    stop_words: set = set(stopwords.words("english"))
    df[prefix + "stopword_num"] = (
        df["text_preprocessed"]
        .str.lower()
        .map(lambda x: len([word for word in x.split() if word in stop_words]))
    )

    df[prefix + "punctuation_num"] = df["text"].map(
        lambda x: len([c for c in str(x) if c in string.punctuation])
    )
    df[prefix + "hashtag_num"] = df["text"].map(
        lambda x: len([c for c in str(x) if c == "#"])
    )
    df[prefix + "mention_num"] = df["text"].map(
        lambda x: len([c for c in str(x) if c == "@"])
    )
    df[prefix + "word_num"] = df["text"].map(lambda x: len(x.split()))
    df[prefix + "unique_word_num"] = df["text"].map(lambda x: len(set(x.split())))
    df[prefix + "avg_word_num"] = df["text"].map(
        lambda x: np.mean([len(word) for word in x.split()])
    )
    df = df.drop(["id", "keyword", "location", "text", "text_preprocessed"], axis=1)
    if "target" in df.columns:
        df = df.drop("target", axis=1)
    return df


class FEBasicTrainData(Tweet):
    def requires(self):
        return dict(train=PreprocessTrainData())

    def run(self):
        logger.info(f"FE Train Data")
        dftrain: pd.DataFrame = self.load("train")
        dftrain = fe_basic(dftrain)
        self.dump(dftrain)


class FEBasicTestData(Tweet):
    def requires(self):
        return dict(test=PreprocessTestData())

    def run(self):
        logger.info(f"FE Test Data")
        dftest: pd.DataFrame = self.load("test")
        dftest = fe_basic(dftest)
        self.dump(dftest)
