# kaggle_disaster_tweets_gokart

My solution for Kaggle Tweet Competition (PB Score: 0.83640) (https://www.kaggle.com/c/nlp-getting-started)

## Ponchi

![Alt text](img/ponchi.png?raw=true "Ponchi")

## Step1. Set up environments with pipenv

```bash
pipenv install --dev --skip-lock
```

## Step2. Download train.csv, test.csv, sample_submission.csv

Put them in ./nlp-getting-started directory.

## Step3. Start up luigi / mlflow server (in other terminal windows)

```bash
luigid
```

```bash
mlflow ui
```

## Step4. Run cross validation

```bash
pipenv run python main.py tweet.CrossValidation
```

## Step5. Create submission file

```bash
pipenv run python main.py tweet.CreateSubmissionFile
```

## Test

```bash
python -m unittest discover -s ./test/unit_test/
```
