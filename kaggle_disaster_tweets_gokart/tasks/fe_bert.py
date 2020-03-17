from kaggle_disaster_tweets_gokart import Tweet, LoadTrainData, LoadTestData
from sentence_transformers import SentenceTransformer, LoggingHandler
import luigi
import pandas as pd
import numpy as np

import logging

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


class FEBertTrain(Tweet):
    bert_model_name = luigi.Parameter(default="bert-base-nli-mean-tokens")

    def requires(self):
        return LoadTrainData()

    def run(self):
        df: pd.DataFrame = self.load()
        embedder = SentenceTransformer(self.bert_model_name)
        embeddings = embedder.encode(df["text"].values.tolist())
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    embeddings,
                    columns=[
                        f"fe_bert_{i}" for i in range(np.array(embeddings).shape[1])
                    ],
                ),
            ],
            axis=1,
        )
        self.dump(df.filter(like="fe_bert_"))


class FEBertTest(Tweet):
    bert_model_name = luigi.Parameter(default="bert-base-nli-mean-tokens")

    def requires(self):
        return LoadTestData()

    def run(self):
        df: pd.DataFrame = self.load()
        embedder = SentenceTransformer(self.bert_model_name)
        embeddings = embedder.encode(df["text"].values.tolist())
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    embeddings,
                    columns=[
                        f"fe_bert_{i}" for i in range(np.array(embeddings).shape[1])
                    ],
                ),
            ],
            axis=1,
        )
        self.dump(df.filter(like="fe_bert_"))
