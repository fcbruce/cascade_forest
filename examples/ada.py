#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Sat 29 Sep 2018 16:16:58
#
#

import lbooster.adaboost as ada
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score as auc



X = np.random.rand(5000, 500)
y = np.array(np.random.rand(5000) > 0.5, dtype=np.float32)
d_train = xgb.DMatrix(X, y)

watchlist = [(d_train, 'train')]

params = {
    "objective": "binary:logistic",
    "booster": "gbtree",
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

ada = ada.train_xgb(10, params, d_train, 50, watchlist)
pred = ada.predict(d_train)
print(auc(y, pred))
