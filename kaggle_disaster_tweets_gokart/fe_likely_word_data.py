from logging import getLogger
from kaggle_disaster_tweets_gokart import (
    Tweet,
    PreprocessTrainData,
    PreprocessTestData,
    MakeTargetAvgWord,
)
import pandas as pd
from tqdm import tqdm

logger = getLogger(__name__)


def fe_likely_word(df: pd.DataFrame, target_avg_word: pd.DataFrame):
    for word in tqdm(
        target_avg_word.query("target_avg > 0.7 and target_count > 3")["word"]
    ):
        df["fe_likely_word_{}".format(word)] = (
            df["text_preprocessed"]
            .str.lower()
            .str.contains(word, regex=False)
            .astype(int)
        )

    for word in tqdm(
        target_avg_word.query("target_avg < 0.3 and target_count > 3")["word"]
    ):
        df["fe_unlikely_word_{}".format(word)] = (
            df["text_preprocessed"]
            .str.lower()
            .str.contains(word, regex=False)
            .astype(int)
        )
    df = pd.concat(
        [df.filter(like="fe_likely_word_"), df.filter(like="fe_unlikely_word_")],
        axis=1,
    )
    return df


class FELikelyWordTrain(Tweet):
    def requires(self):
        return dict(df=PreprocessTrainData(), target_avg_word=MakeTargetAvgWord())

    def run(self):
        df: pd.DataFrame = self.load("df")
        target_avg_word: pd.DataFrame = self.load("target_avg_word")

        logger.info("creating likely word features of train")
        df = fe_likely_word(df, target_avg_word)
        self.dump(df)


class FELikelyWordTest(Tweet):
    def requires(self):
        return dict(df=PreprocessTestData(), target_avg_word=MakeTargetAvgWord())

    def run(self):
        df: pd.DataFrame = self.load("df")
        target_avg_word: pd.DataFrame = self.load("target_avg_word")

        logger.info("creating likely word features of test")
        df = fe_likely_word(df, target_avg_word)
        self.dump(df)
