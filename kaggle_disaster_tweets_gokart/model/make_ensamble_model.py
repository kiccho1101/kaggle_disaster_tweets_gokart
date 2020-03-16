from logging import getLogger

from lightgbm import LGBMClassifier

from kaggle_disaster_tweets_gokart.base import Tweet
from kaggle_disaster_tweets_gokart.model.ensamble import EnsambleModel, Ensamble
from kaggle_disaster_tweets_gokart.model.lgbm import OptimizeLGBMHP

logger = getLogger(__name__)


class MakeEnsambleModel(Tweet):
    def requires(self):
        return dict(lgbm_params=OptimizeLGBMHP())

    def run(self):
        ensamble: Ensamble = Ensamble(
            [
                EnsambleModel(
                    LGBMClassifier(**self.load("lgbm_params")["best_params"]), 1.0, {}
                )
            ]
        )
        self.dump(ensamble)
