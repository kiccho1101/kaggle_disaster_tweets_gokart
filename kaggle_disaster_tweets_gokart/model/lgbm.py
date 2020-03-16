from logging import getLogger

import luigi
from luigi.util import inherits
import redshells
from kaggle_disaster_tweets_gokart.tasks.make_feature_data import MakeTrainFeatureData
from kaggle_disaster_tweets_gokart.base import Tweet

logger = getLogger(__name__)


@inherits(MakeTrainFeatureData)
class OptimizeLGBMHP(Tweet):
    def requires(self):
        df = self.clone(MakeTrainFeatureData)
        return redshells.train.OptimizeBinaryClassificationModel(
            rerun=True,
            train_data_task=df,
            target_column_name="target",
            model_name="LGBMClassifier",
            test_size=0.2,
            optuna_param_name="LGBMClassifier_default",
        )

    def run(self):
        model = self.load()
        self.dump(model)
