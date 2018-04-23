#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Sat 21 Apr 2018 15:41:51
#
#

import numpy as np
import xgboost as xgb
import lightgbm as lgb

class Dataset(object):

    def __init__(self, X, y=None, feature_name=None, shuffle=False):
        # TODO support pandas dataframe
        if isinstance(X, np.ndarray):
            self.X = X
            self.n_sample = X.shape[0]
            self.n_col = X.shape[1]
        else:
            raise TypeError("X is not numpy.array")
        if y is not None:
            if (isinstance(y, list) or isinstance(y, np.ndarray) and len(y.shape) == 1) and len(y) == self.n_sample:
                self.y = np.array(y)
            else:
                raise TypeError("y is not list or numpy 1d array or shape error")
        else:
            self.y = None

        if feature_name is not None:
            if isinstance(feature_name, list) and len(feature_name) == self.n_col:
                self.feature_name = feature_name
            else:
                raise TypeError("feature_name is not list or shape error")

        if shuffle:
            idx = np.arange(self.n_sample)
            np.random.shuffle(idx)
            self.X = self.X[idx, :]
            self.y = self.y[idx]

        self.attach = None


    def set_attach(self, attach):

        self.attach = attach


    def clear_attach(self):

        self.attach = None


    def kfold_data(self, kfold=1, kth=0):

        arange = np.arange(self.n_sample)
        cv = arange % kfold == kth
        idx = ~cv
        X_train = self.X[idx, :] if self.attach is None else np.hstack((self.attach[idx, :], self.X[idx, :]))
        X_cv = self.X[cv, :] if self.attach is None else np.hstack((self.attach[cv, :], self.X[cv, :]))
        return (X_train, self.y[idx]), (X_cv, self.y[cv])


    def data(self):
        X = self.X if self.attach is None else np.hstack((self.attach, self.X))
        return X


    def label(self):
        return self.y


class XGBoostForest(object):

    def __init__(self, kfold, kth, model_file=None):

        self.kfold = kfold
        self.kth = kth


    def train(self, params, num_round, d_train, d_test=None):

        (X_train, y_train), (X_cv, y_cv) = d_train.kfold_data(self.kfold, self.kth)
        d_train = xgb.DMatrix(X_train, y_train)
        d_cv = xgb.DMatrix(X_cv, y_cv)
        X_test, y_test = d_test.data()
        d_test = xgb.DMatrix(X_test, y_test)

        watch_list = [(d_train, 'train'), (d_cv, 'cv')]
        self.bst = xgb.train(params, d_train, num_round, watch_list)

        pred_cv = self.bst.predict(d_cv)
        pred_test = self.bst.predict(d_test)

        return pred_cv, pred_test


    def predict(self, data):

        if self.bst is None:
            raise ValueError('bst is None')

        X, y = data.data()
        d = xgb.DMatrix(X, y)

        return self.bst.predict(d)

class LightGBMForest(object):

    def __init__(self, kfold, kth, model_file=None):

        self.kfold = kfold
        self.kth = kth


    def train(self, params, num_round, d_train, d_test=None):

        (X_train, y_train), (X_cv, y_cv) = d_train.kfold_data(self.kfold, self.kth)
        d_train = lgb.Dataset(X_train, y_train)
        d_cv = lgb.Dataset(X_cv, y_cv, reference=d_train)
        X_test, y_test = d_test.data(), d_test.label()

        watch_list = [d_train,  d_cv]
        self.bst = lgb.train(params, d_train, num_boost_round=num_round, valid_sets=watch_list)

        pred_cv = self.bst.predict(X_cv)
        pred_test = self.bst.predict(X_test)

        return pred_cv, pred_test


    def predict(self, data):

        if self.bst is None:
            raise ValueError('bst is None')

        X = data.data()

        return self.bst.predict(X)



class CascadeForest(object):

    def __init__(self, config=None, model_file=None):

        self.models = {}

        if model_file is not None:
            pass

        if config is not None:
            self.max_layer = self._get_cfg_value(config, 'max_layer', None, True, int)
            self.forests = self._get_cfg_value(config, 'forests', None, True, list)


    def train(self, config, d_train, d_test=None):

        last_train_pred = None
        last_test_pred = None

        for layer in range(self.max_layer):
            layer_train_pred = []
            layer_test_pred = []

            for fi, forest_cfg in enumerate(self.forests):
                lib = self._get_cfg_value(forest_cfg, 'lib', None, True, str)
                kfold = self._get_cfg_value(forest_cfg, 'kfold', None, True, int)
                num_round = self._get_cfg_value(forest_cfg, 'num_round', None, True, int)
                params = self._get_cfg_value(forest_cfg, 'params', None, True, dict)

                if lib == 'xgb':
                    forest = XGBoostForest
                elif lib == 'lgb':
                    forest = LightGBMForest
                else:
                    raise ValueError('do not support {}'.format(lib))

                cur_train_pred = np.zeros(d_train.n_sample)
                cur_test_pred = np.zeros(d_test.n_sample)
                for kth in range(kfold):
                    bst = forest(kfold, kth)
                    self.models[(layer, fi, kth)] = bst
                    pred_cv, pred_test = bst.train(params, num_round, d_train, d_test)

                    cur_train_pred[range(kth, d_train.n_sample, kfold)] = pred_cv

                    cur_test_pred += pred_test
                
                cur_test_pred /= kfold

                layer_train_pred.append(cur_train_pred)
                layer_test_pred.append(cur_test_pred)

            last_train_pred = np.array(layer_train_pred).T
            last_test_pred = np.array(layer_test_pred).T
            d_train.set_attach(last_train_pred)
            d_test.set_attach(last_test_pred)

        d_train.clear_attach()
        d_test.clear_attach()

        return last_train_pred, last_test_pred


    def predict(self, data):

        if not isinstance(data, Dataset):
            raise TypeError('data shoud be Dataset, but {} found'.format(type(data)))

        for layer in range(self.max_layer):
            layer_pred = []

            for fi, forest_cfg in enumerate(self.forests):
                kfold = self._get_cfg_value(forest_cfg, 'kfold', None, True, int)

                cur_pred = np.zeros(data.n_sample)
                for kth in range(kfold):
                    pred = self.models[(layer, fi, kth)].predict(data)
                    cur_pred += pred

                cur_pred /= kfold

                layer_pred.append(cur_pred)

            last_pred = np.array(layer_pred).T
            data.set_attach(last_pred)

        return last_pred.mean(axis=1)


    def _get_cfg_value(self, cfg, key, default, required=False, value_type=None):

        value = cfg.get(key, default)
        if required and value is None:
            raise ValueError('{} should not be None'.format(key))
        if value_type is not None and value is not None:
            if not isinstance(value, value_type):
                raise TypeError('{} should be {}, but is {}'.format(key, value_type, type(value)))

        return value
