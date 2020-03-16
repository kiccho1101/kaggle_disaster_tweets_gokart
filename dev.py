# %%
from IPython import get_ipython
import sys


def ipy_exit(*args):
    exit(keep_kernel=True)


ipython = get_ipython()
# ipython.magic("load_ext autoreload")
# ipython.magic("autoreload 2")
sys.exit = ipy_exit

import gokart
import luigi
import numpy as np
import pandas as pd

import kaggle_disaster_tweets_gokart

luigi.configuration.LuigiConfigParser.add_config_path("./conf/param.ini")
np.random.seed(42)


# %%
gokart.run(["tweet.FELikelyWordTest", "--rerun"])


# %%
from thunderbolt import Thunderbolt

tb = Thunderbolt("./resource")
tb.get_task_df()

# %%
tb.get_data("MakeTrainFeatureData")


# %%
