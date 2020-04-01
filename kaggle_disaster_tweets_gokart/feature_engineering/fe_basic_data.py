from logging import getLogger

import pandas as pd

import re
import regex
import swifter
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from kaggle_disaster_tweets_gokart import Tweet, PreprocessTrainData, PreprocessTestData

logger = getLogger(__name__)


def get_regex_num(pattern: str, text: str):
    return len(regex.compile(pattern).findall(text))


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

    df[prefix + "number_num"] = df["text_preprocessed"].swifter.apply(
        lambda x: get_regex_num(r"\<NUMBER\>", x)
    )
    df[prefix + "user_num"] = df["text_preprocessed"].swifter.apply(
        lambda x: get_regex_num(r"\<USER\>", x)
    )
    df[prefix + "earthquake_num"] = (
        df["text_preprocessed"]
        .str.lower()
        .swifter.apply(lambda x: get_regex_num(r"earthquake", x))
    )
    df[prefix + "flood_num"] = (
        df["text_preprocessed"]
        .str.lower()
        .swifter.apply(lambda x: get_regex_num(r"flood", x))
    )
    df[prefix + "face_num"] = df["text_preprocessed"].swifter.apply(
        lambda x: get_regex_num(r"FACE\>", x)
    )
    df[prefix + "repeat_num"] = df["text_preprocessed"].swifter.apply(
        lambda x: get_regex_num(r"\<REPEAT\>", x)
    )
    for char in list("abcdefghijklmnopqrstuvwxyz"):
        df[prefix + f"{char}_num"] = (
            df["text"].str.lower().swifter.apply(lambda x: get_regex_num(char, x))
        )

    df = df.filter(like="fe_basic_")
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
