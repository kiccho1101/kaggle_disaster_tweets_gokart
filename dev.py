# %%
from IPython import get_ipython
import sys


def ipy_exit(*args):
    exit(keep_kernel=True)


ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")
sys.exit = ipy_exit

import gokart
import luigi
import numpy as np
import pandas as pd

import kaggle_disaster_tweets_gokart

luigi.configuration.LuigiConfigParser.add_config_path("./conf/param.ini")
np.random.seed(42)


# %%
# gokart.run(["tweet.MakeEnsembleModel", "--rerun"])


# %%
from thunderbolt import Thunderbolt

tb = Thunderbolt("./resource")
tb.get_task_df()


# %%
import numpy as np

# %%
import re
import regex
import swifter

df: pd.DataFrame = tb.get_data("MakeTrainSelectedFeatureData")


# %%


def plot_fi(tb):

    fi = tb.get_data("MakeFeatureImportance")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 12))
    fi.set_index("feature")["importance"].iloc[710:780].plot(kind="barh")
    plt.show()


plot_fi(tb)


# %%
