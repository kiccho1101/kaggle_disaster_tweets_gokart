from logging import getLogger
import pandas as pd
from kaggle_disaster_tweets_gokart.base import Tweet


logger = getLogger(__name__)


class LoadTrainData(Tweet):
    def run(self):
        df = pd.read_csv("./nlp-getting-started/train.csv")
        self.dump(df)


class LoadTestData(Tweet):
    def run(self):
        df = pd.read_csv("./nlp-getting-started/test.csv")
        self.dump(df)
