from logging import getLogger

import luigi
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import sklearn.ensemble
import sklearn.linear_model
import sklearn.svm

from kaggle_disaster_tweets_gokart import Tweet

logger = getLogger(__name__)


class MakeEnsembleModel(Tweet):
    models = luigi.ListParameter()
    lgbm_params = luigi.DictParameter()
    xgb_params = luigi.DictParameter()
    logistic_params = luigi.DictParameter()
    sgd_params = luigi.DictParameter()
    svc_params = luigi.DictParameter()

    def run(self):

        all_estimators = {
            "lgbm": LGBMClassifier(**self.lgbm_params),
            "xgb": XGBClassifier(**self.xgb_params),
            "logistic": sklearn.linear_model.LogisticRegression(**self.logistic_params),
            "sgd": sklearn.linear_model.SGDClassifier(**self.sgd_params),
            "svc": sklearn.svm.LinearSVC(**self.svc_params),
        }

        estimators = [(n, m) for n, m in all_estimators.items() if n in self.models]

        model = sklearn.ensemble.StackingClassifier(estimators=estimators, n_jobs=-1,)
        self.dump(model)
