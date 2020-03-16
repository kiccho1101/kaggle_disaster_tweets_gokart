from logging import getLogger

import pandas as pd
from typing import List
import luigi
from kaggle_disaster_tweets_gokart import (
    Tweet,
    PreprocessTrainData,
    PreprocessTestData,
    FEBasicTrainData,
    FEBasicTestData,
    FELikelyWordTrain,
    FELikelyWordTest,
)


logger = getLogger(__name__)


class MakeTrainFeatureData(Tweet):
    select_cols: List[str] = luigi.ListParameter()

    def requires(self):
        return dict(
            pp_df=PreprocessTrainData(),
            fe_basic_df=FEBasicTrainData(),
            fe_likely_word_df=FELikelyWordTrain(),
        )

    def run(self):
        df: pd.DataFrame = pd.concat(
            [
                self.load("pp_df")["target"],
                self.load("fe_basic_df"),
                self.load("fe_likely_word_df"),
            ],
            axis=1,
        )
        selected_cols: List[str] = ["target"]
        for select_col in self.select_cols:
            selected_cols += [col for col in df.columns if col.startswith(select_col)]
        selected_cols = list(set(selected_cols))
        self.dump(df[selected_cols])


class MakeTestFeatureData(Tweet):
    select_cols: List[str] = luigi.ListParameter()

    def requires(self):
        return dict(
            pp_df=PreprocessTestData(),
            fe_basic_df=FEBasicTestData(),
            fe_likely_word_df=FELikelyWordTest(),
        )

    def run(self):
        df: pd.DataFrame = pd.concat(
            [self.load("fe_basic_df"), self.load("fe_likely_word_df")], axis=1
        )
        selected_cols: List[str] = []
        for select_col in self.select_cols:
            selected_cols += [col for col in df.columns if col.startswith(select_col)]
        selected_cols = list(set(selected_cols))
        self.dump(df[selected_cols])
