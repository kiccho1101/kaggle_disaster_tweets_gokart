import pandas as pd
import sklearn.feature_extraction
from kaggle_disaster_tweets_gokart import Tweet, PreprocessTrainData, PreprocessTestData


class MakeTfidfModel(Tweet):
    def requires(self):
        return dict(dftrain=PreprocessTrainData(), dftest=PreprocessTestData())

    def run(self):
        tfidf = sklearn.feature_extraction.text.TfidfVectorizer(
            stop_words="english", min_df=4
        )
        tfidf.fit(
            pd.concat([self.load("dftrain"), self.load("dftest"),], axis=0)[
                "text_preprocessed"
            ]
        )
        self.dump(tfidf)


class FETfidfTrain(Tweet):
    def requires(self):
        return dict(df=PreprocessTrainData(), tfidf=MakeTfidfModel())

    def run(self):
        df = self.load("df")
        tfidf = self.load("tfidf")
        result = pd.DataFrame(tfidf.transform(df["text_preprocessed"]).toarray())
        result.columns = [f"fe_tfidf_{i}" for i in range(result.shape[1])]
        self.dump(result)


class FETfidfTest(Tweet):
    def requires(self):
        return dict(df=PreprocessTestData(), tfidf=MakeTfidfModel())

    def run(self):
        df = self.load("df")
        tfidf = self.load("tfidf")
        result = pd.DataFrame(tfidf.transform(df["text_preprocessed"]).toarray())
        result.columns = [f"fe_tfidf_{i}" for i in range(result.shape[1])]
        self.dump(result)
