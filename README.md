# kaggle_disaster_tweets_gokart

My solution for Kaggle Tweet Competition (https://www.kaggle.com/c/nlp-getting-started)

## Step1. Set up environments with pipenv

```bash
pipenv install --dev --skip-lock
```

## Step2. Download train.csv, test.csv

Put them in ./nlp-getting-started directory.

## Step3. Start up luigi server (in other terminal window)

```bash
luigid
```

## Step4. Run the code

```bash
python main.py tweet.CrossValidation
```

## Test

```bash
python -m unittest discover -s ./test/unit_test/
```
