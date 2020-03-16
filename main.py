import luigi
import numpy as np
import gokart

import kaggle_disaster_tweets_gokart

if __name__ == "__main__":
    luigi.configuration.LuigiConfigParser.add_config_path("./conf/param.ini")
    np.random.seed(57)
    gokart.run()
