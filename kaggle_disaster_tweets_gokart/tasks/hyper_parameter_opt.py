import redshells
from luigi.util import inherits
from kaggle_disaster_tweets_gokart import Tweet, MakeTrainSelectedFeatureData


@inherits(MakeTrainSelectedFeatureData)
class OptimizeLGBMHP(Tweet):
    def requires(self):
        df = self.clone(MakeTrainSelectedFeatureData)
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


@inherits(MakeTrainSelectedFeatureData)
class OptimizeXGBHP(Tweet):
    def requires(self):
        df = self.clone(MakeTrainSelectedFeatureData)
        return redshells.train.OptimizeBinaryClassificationModel(
            rerun=True,
            train_data_task=df,
            target_column_name="target",
            model_name="XGBClassifier",
            test_size=0.2,
            optuna_param_name="XGBClassifier_default",
        )

    def run(self):
        model = self.load()
        self.dump(model)
