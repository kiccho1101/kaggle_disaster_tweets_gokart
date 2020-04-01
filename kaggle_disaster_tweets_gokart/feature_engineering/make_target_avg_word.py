from logging import getLogger

from tqdm import tqdm
import pandas as pd
from typing import List

from kaggle_disaster_tweets_gokart import Tweet, PreprocessTrainData

logger = getLogger(__name__)


class MakeTargetAvgWord(Tweet):
    def requires(self):
        return PreprocessTrainData()

    def run(self):
        df: pd.DataFrame = self.load()
        corpus: List[str] = df["text_preprocessed"].str.lower().str.split(
            expand=True
        ).unstack().unique()
        te_words = []
        logger.info("creating target_avg_word")
        for word in tqdm(corpus):
            if word is not None:
                df_part = df[
                    df["text_preprocessed"].str.lower().str.contains(word, regex=False)
                ]
                te_words.append(
                    {
                        "word": word,
                        "target_avg": df_part["target"].mean(),
                        "target_count": len(df_part),
                    }
                )
        target_avg_df = pd.DataFrame(te_words)
        self.dump(target_avg_df)
