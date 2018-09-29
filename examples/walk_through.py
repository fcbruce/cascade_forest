#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Sat 21 Apr 2018 18:09:37
#
#


import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

import lbooster.cascade_forest as cf


X = np.random.rand(5000, 500)
y = np.array(np.random.rand(5000) > 0.5, dtype=np.float32)
d_train = cf.Dataset(X, y)

X_test = np.random.rand(50, 500)
y_test = np.array(np.random.rand(50) > 0.5, dtype=np.float32)
d_test = cf.Dataset(X_test, y_test)

config = {
        "max_layer": 3,
        "forests": [
            {
                "lib": "xgb", "kfold": 5, "num_round": 20,
                "params": {
                    "objective": "binary:logistic",
                    "booster": "gbtree",
                    "tree_method": "hist",
                    "eta": 0.5,
                    "max_depth": 5,
                    "num_leaves": 127,
                    "eval_metric": ["auc"],
                    "min_child_weight": 50,
                    "colsample_bytree": 0.5,
                    "subsample": 0.5,
                    "gamma": 0.17,
                    "alpha": 0.1,
                    "lambda": 0.818,
                    "num_threads": 4
                    }

                },
            {
                "lib": "lgb", "kfold": 5, "num_round": 20,
                "params": {

                    "task": "train",
                    "objective": "binary",
                    "boost": "gbdt",
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "metric": ["auc"],
                    "min_sum_hessian_in_leaf": 60,
                    "feature_fraction": 0.2,
                    "bagging_fraction": 0.5,
                    "bagging_freq": 1,
                    "min_gain_to_split": 0.07,
                    "lambda_l1": 0.1,
                    "lambda_l2": 0.618,
                    "num_threads": 4
                    }
                }
            ]
        }

cas = cf.CascadeForest(config)


cas.train({'feval': 'auc'}, d_train, d_test)

y_pred = cas.predict(d_test).mean(axis=1)

dirname = 'cas_model'

cas.save_model(dirname)

cas_1 = cf.CascadeForest(dirname=dirname)
y_pred_1 = cas_1.predict(d_test).mean(axis=1)
