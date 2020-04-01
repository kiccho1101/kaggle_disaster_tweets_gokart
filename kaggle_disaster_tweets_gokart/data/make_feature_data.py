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
    FETEKeywordTrain,
    FETEKeywordTest,
    FEOHEKeywordTrain,
    FEOHEKeywordTest,
    FELikelyWordTrain,
    FELikelyWordTest,
    FEBertTrain,
    FEBertTest,
    FETfidfTrain,
    FETfidfTest,
)


logger = getLogger(__name__)


class MakeTrainFeatureData(Tweet):
    def requires(self):
        return dict(
            pp_df=PreprocessTrainData(),
            fe_basic_df=FEBasicTrainData(),
            fe_te_keyword_df=FETEKeywordTrain(),
            fe_ohe_keyword_df=FEOHEKeywordTrain(),
            fe_likely_word_df=FELikelyWordTrain(),
            fe_bert_df=FEBertTrain(),
            fe_tfidf_df=FETfidfTrain(),
        )

    def run(self):
        df: pd.DataFrame = pd.concat(
            [
                self.load("pp_df")["target"],
                self.load("fe_basic_df"),
                self.load("fe_te_keyword_df"),
                self.load("fe_likely_word_df"),
                self.load("fe_bert_df"),
                self.load("fe_tfidf_df"),
            ],
            axis=1,
        )
        self.dump(df)


class MakeTestFeatureData(Tweet):
    def requires(self):
        return dict(
            pp_df=PreprocessTestData(),
            fe_basic_df=FEBasicTestData(),
            fe_te_keyword_df=FETEKeywordTest(),
            fe_ohe_keyword_df=FEOHEKeywordTest(),
            fe_likely_word_df=FELikelyWordTest(),
            fe_bert_df=FEBertTest(),
            fe_tfidf_df=FETfidfTest(),
        )

    def run(self):
        df: pd.DataFrame = pd.concat(
            [
                self.load("fe_basic_df"),
                self.load("fe_te_keyword_df"),
                self.load("fe_likely_word_df"),
                self.load("fe_bert_df"),
                self.load("fe_tfidf_df"),
            ],
            axis=1,
        )
        self.dump(df)
