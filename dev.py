# %%
from IPython import get_ipython
import sys

ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")


def ipy_exit(*args):
    exit(keep_kernel=True)


sys.exit = ipy_exit


# %%
import gokart
import luigi
import numpy as np

import kaggle_disaster_tweets_gokart

luigi.configuration.LuigiConfigParser.add_config_path("./conf/param.ini")
np.random.seed(42)
gokart.run(["kaggle_disaster_tweets_gokart.Sample", "--local-scheduler"])

# %%
