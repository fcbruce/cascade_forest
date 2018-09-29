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
from sklearn.metrics import roc_auc_score
import os.path as osp
import json
import shutil
import os

import logging
logger = logging.getLogger(__name__)

def set_logger_level(level):
    logger.setLevel(level)


fevals_switcher = {
        'auc': roc_auc_score
        }

def get_feval(feval):

    return fevals_switcher.get(feval)


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

        if model_file is not None:
            self.bst = xgb.Booster(model_file=model_file)

    def train(self, params, num_round, d_train, d_test=None):

        (X_train, y_train), (X_cv, y_cv) = d_train.kfold_data(self.kfold, self.kth)
        d_train = xgb.DMatrix(X_train, y_train)
        d_cv = xgb.DMatrix(X_cv, y_cv)
        X_test, y_test = d_test.data(), d_test.label()
        d_test = xgb.DMatrix(X_test, y_test)

        watch_list = [(d_train, 'train'), (d_cv, 'cv'), (d_test, 'test')]
        self.bst = xgb.Booster(model_file=xgb.train(params, d_train, num_round, watch_list).save_raw())

        pred_cv = self.bst.predict(d_cv)
        pred_test = self.bst.predict(d_test)

        return pred_cv, pred_test


    def predict(self, data):

        if self.bst is None:
            raise ValueError('bst is None')

        X = data.data()
        d = xgb.DMatrix(X)

        return self.bst.predict(d)


    def save_model(self, fname):

        if self.bst is None:
            raise Error('this model don\'t need to save {}'.format(fname))

        self.bst.save_model(fname)

class LightGBMForest(object):

    def __init__(self, kfold, kth, model_file=None):

        self.kfold = kfold
        self.kth = kth

        if model_file is not None:
            self.bst = lgb.Booster(model_file=model_file)


    def train(self, params, num_round, d_train, d_test=None):

        (X_train, y_train), (X_cv, y_cv) = d_train.kfold_data(self.kfold, self.kth)
        d_train = lgb.Dataset(X_train, y_train)
        d_cv = lgb.Dataset(X_cv, y_cv, reference=d_train)
        X_test, y_test = d_test.data(), d_test.label()
        d_test = lgb.Dataset(X_test, y_test, reference=d_train)

        watch_list = [d_train,  d_cv, d_test]
        self.bst = lgb.train(params, d_train, num_boost_round=num_round, valid_sets=watch_list)

        pred_cv = self.bst.predict(X_cv)
        pred_test = self.bst.predict(X_test)

        return pred_cv, pred_test


    def predict(self, data):

        if self.bst is None:
            raise ValueError('bst is None')

        X = data.data()

        return self.bst.predict(X)


    def save_model(self, fname):

        if self.bst is None:
            raise Error('this model don\'t need to save {}'.format(fname))

        self.bst.save_model(fname)


_MODEL_FILE_TEMPLATE = '{}_layer-{}_forest-{}_fold-{}.model'

class CascadeForest(object):

    def __init__(self, config=None, dirname=None):

        self.models = {}

        if dirname is not None:
            self.load_model(dirname)
            return

        if config is not None:
            self.max_layer = self._get_cfg_value(config, 'max_layer', None, True, int)
            self.forests = self._get_cfg_value(config, 'forests', None, True, list)


    def train(self, config, d_train, d_test=None):

        feval_name = self._get_cfg_value(config, 'feval', None, False, str)
        feval = get_feval(feval_name)

        last_train_pred = None
        last_test_pred = None

        arange = np.arange(d_train.n_sample)
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

                    cv = arange % kfold == kth
                    cur_train_pred[cv] = pred_cv

                    cur_test_pred += pred_test
                
                cur_test_pred /= kfold

                if feval is not None:
                    eval_cv = feval(d_train.label(), cur_train_pred)
                    eval_test = feval(d_test.label(), cur_test_pred)

                    logger.info('layer-%d forest-%d, train-%s: %f, test-%s: %f', layer, fi, feval_name, eval_cv, feval_name, eval_test)


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

        data.clear_attach()

        return last_pred


    def save_model(self, dirname):

        if osp.isdir(dirname): 
            shutil.rmtree(dirname)
        elif osp.exists(dirname):
            os.remove(dirname)

        os.mkdir(dirname)

        with open(osp.join(dirname, 'cascade_forest.json'), 'w') as f:
            cfg = {
                    'max_layer': self.max_layer,
                    'forests': self.forests
                    }
            json.dump(cfg, f)

        for layer in range(self.max_layer):
            for fi, forest_cfg in enumerate(self.forests):
                lib = self._get_cfg_value(forest_cfg, 'lib', None, True, str)
                kfold = self._get_cfg_value(forest_cfg, 'kfold', None, True, int)

                for kth in range(kfold):
                    self.models[(layer, fi, kth)].save_model(osp.join(dirname, _MODEL_FILE_TEMPLATE.format(lib, layer, fi, kth)))

    def load_model(self, dirname):

        with open(osp.join(dirname, 'cascade_forest.json'), 'r') as f:
            cfg = json.load(f)
            self.max_layer = self._get_cfg_value(cfg, 'max_layer', None, True, int)
            self.forests = self._get_cfg_value(cfg, 'forests', None, True, list)

        for layer in range(self.max_layer):
            for fi, forest_cfg in enumerate(self.forests):
                lib = self._get_cfg_value(forest_cfg, 'lib', None, True, str)
                kfold = self._get_cfg_value(forest_cfg, 'kfold', None, True, int)

                if lib == 'xgb':
                    forest = XGBoostForest
                elif lib == 'lgb':
                    forest = LightGBMForest
                else:
                    raise ValueError('do not support {}'.format(lib))

                for kth in range(kfold):
                    self.models[(layer, fi, kth)] = forest(kfold, kth, (osp.join(dirname, _MODEL_FILE_TEMPLATE.format(lib, layer, fi, kth))))


    def _get_cfg_value(self, cfg, key, default, required=False, value_type=None):

        value = cfg.get(key, default)
        if required and value is None:
            raise ValueError('{} should not be None'.format(key))
        if value_type is not None and value is not None:
            if not isinstance(value, value_type):
                raise TypeError('{} should be {}, but is {}'.format(key, value_type, type(value)))

        return value
