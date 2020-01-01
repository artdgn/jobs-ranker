import abc
import itertools

import numpy as np
import pandas as pd
import scipy.stats
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from jobs_ranker.config import common
from jobs_ranker.utils.instrumentation import LogCallsTimeAndOutput
from jobs_ranker.utils.logger import logger

SPEARMAN = 'spearman'
R2 = 'r2'
MAIN_METRIC = SPEARMAN


class FunctionTransformerFeatNames(FunctionTransformer, LogCallsTimeAndOutput):

    def __init__(self, func=None, name='', validate=False):
        super().__init__(func=func, validate=validate)
        self._name = name

    def get_feature_names(self):
        return [self._name]


class PipelineFeatNames(Pipeline, LogCallsTimeAndOutput):

    def get_feature_names(self):
        return [n for n in self.steps[-1][1].get_feature_names()]


class RegPipelineBase(abc.ABC, LogCallsTimeAndOutput):

    def __init__(self, text_cols, num_cols):
        super().__init__()
        self.text_cols = text_cols
        self.num_cols = num_cols

        self.reg = self._reg()

        self.transformer = FeatureUnion([
            *(('tfidf_' + col, self._tfidf_pipe(col)) for col in self.text_cols),
            *(('noop_' + col, self._noop_pipe(col)) for col in self.num_cols),
        ])

        self.pipe = PipelineFeatNames([
            ('transformer', self.transformer),
            ('regressor', self.reg)])

    @staticmethod
    @abc.abstractmethod
    def _reg():
        return RegressorMixin()

    @staticmethod
    def _tfidf_pipe(col):
        return PipelineFeatNames([
            ('extract_docs', FunctionTransformerFeatNames(
                lambda x: x[col].values, name=col, validate=False)),
            ('tfidf_' + col, TfidfVectorizer(
                ngram_range=common.MLParams.reg_tfidf_ngram_range,
                min_df=common.MLParams.reg_tfidf_min_df,
                stop_words='english'))])

    @staticmethod
    def _noop_pipe(col):
        return PipelineFeatNames([
            ('noop_' + col, FunctionTransformerFeatNames(
                lambda x: x[col].values.reshape(-1, 1),
                name=col,
                validate=False))])

    def _train_eval(self, x, y, test_ratio, target_name='', baselines=()):
        # split
        x_train, x_test, y_train, y_test, _, test_mask = train_test_split(
            x, y, np.arange(len(y)),
            test_size=test_ratio or common.MLParams.test_ratio,
            shuffle=common.MLParams.shuffle_split
        )
        self.pipe.fit(x_train, y_train)
        y_pred = self.predict(x_test)

        metrics = self.print_metrics(y_test, y_pred, target_name=target_name)

        return metrics, self._score_baselines(baselines, test_mask, y_test)

    def _score_baselines(self, baselines, test_mask, y_true):
        return [score_metrics(y_true, vec[test_mask]) for vec in baselines]

    def _refit(self, x, y):
        self.pipe.fit(x, y)

    def predict(self, X, **kwargs):
        return self.pipe.predict(X, **kwargs)

    def print_metrics(self, y_test, y_pred, target_name):
        metrics = score_metrics(y_test, y_pred)
        logger.info(f"\n {pd.Series(metrics).to_frame(f'{target_name} :').transpose()}")
        binary_metrics = binary_scores(y_test, y_pred)
        if binary_metrics is not None:
            logger.info(f"{target_name}, binary scores:\n {binary_metrics}")
        return metrics

    def train_eval(self, df, y_col, test_ratio=None, target_name='', baselines=()):
        """
        :param target_name: for printouts
        :param baselines: baseline predictors to be scored on the same test-set
        :return: model and baseline score metrics
        """

        if df[y_col].isnull().sum():
            raise ValueError('Target column contains nans')

        x, y = df[self.text_cols + self.num_cols], df[y_col].values

        metrics, baselines_metrics = self._train_eval(x, y,
                                                      test_ratio=test_ratio,
                                                      target_name=target_name,
                                                      baselines=baselines)

        self.print_top_n_features(
            x, y, n=common.InfoParams.top_n_feat, target_name=target_name)

        self._refit(x, y)

        return metrics, baselines_metrics

    @classmethod
    def exhaustive_column_selection(cls, text_cols, num_cols, x, y, metric, test_ratio):
        res = []

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=test_ratio or common.MLParams.test_ratio,
            shuffle=common.MLParams.shuffle_split)

        for cols in all_subsets(text_cols + num_cols):
            this = cls(
                [col for col in text_cols if col in cols],
                [col for col in num_cols if col in cols])

            this.pipe.fit(x_train[list(cols)], y_train)
            y_pred = this.pipe.predict(x_test[list(cols)])
            test_metrics = score_metrics(y_test, y_pred)

            res.append((test_metrics[metric], cols))

            logger.info(
                f'selection {test_metrics} {(test_metrics[metric], cols)}')

        best_cols = sorted(res)[-1][1]
        logger.info(f'best: {best_cols}')

        return cls(
            [col for col in text_cols if col in best_cols],
            [col for col in num_cols if col in best_cols])

    def print_top_n_features(self, x, y, n=30, target_name=''):
        # names
        if not hasattr(self.reg, 'feature_importances_'):
            logger.error(f"regressor {self.reg} doesn't have 'feature_importances_' attribute")
            return
        top_n_feat = np.argsort(self.reg.feature_importances_)[-n:]
        feat_names = self.transformer.get_feature_names()
        top_names = np.array(feat_names)[top_n_feat]

        # correlations
        x = self.transformer.transform(x)
        top_feat_x = x[:, top_n_feat].toarray()
        cors_mat, _ = scipy.stats.spearmanr(top_feat_x, y.reshape(-1, 1))
        cors_vec = cors_mat[-1, 0:-1]
        non_zeros = top_feat_x.astype(bool).sum(0)
        df = pd.DataFrame(
            {'name': top_names,
             'correlation': cors_vec,
             'nonzeros': non_zeros}). \
            sort_values('correlation', ascending=False)

        logger.info(f'Top {n} informative features and correlations to '
                    f'{target_name}: \n{df}')


class RFPipeline(RegPipelineBase):

    @staticmethod
    def _reg():
        return RandomForestRegressor(
            n_estimators=common.MLParams.rf_n_estimators, oob_score=True, n_jobs=-1)

    def _train_eval(self, x, y, test_ratio, target_name='', baselines=()):
        if not test_ratio:
            # use OOB scores instead test and refit
            self.pipe.fit(x, y)
            return self.print_metrics(y, self.reg.oob_prediction_, target_name=target_name)
        else:
            return super()._train_eval(x, y, test_ratio=test_ratio,
                                       target_name=target_name, baselines=baselines)


class LGBPipeline(RegPipelineBase):
    class LGBMRegEarlyStop(LGBMRegressor):

        def fit(self, X, y, early_stopping=False):
            if early_stopping:
                x_train, x_valid, y_train, y_valid = train_test_split(
                    X, y, test_size=0.3,
                    shuffle=common.MLParams.shuffle_split)

                LGBMRegressor.fit(self, x_train, y_train,
                                  eval_metric='l2',
                                  early_stopping_rounds=200,
                                  eval_set=(x_valid, y_valid),
                                  verbose=False)

                logger.info(f'LGBM early stopping: '
                            f'setting n_estimators to best_iteration_({self.best_iteration_})')
                self.n_estimators = self.best_iteration_
            return LGBMRegressor.fit(self, X, y)

    @staticmethod
    def _reg():
        return LGBPipeline.LGBMRegEarlyStop(
            n_estimators=common.MLParams.lgbm_max_n_estimators,
            learning_rate=common.MLParams.lgbm_learning_rate)

    def _train_eval(self, x, y, test_ratio, target_name='', baselines=()):
        x_train, x_test, y_train, y_test, _, test_mask = train_test_split(
            x, y, np.arange(len(y)),
            test_size=test_ratio or common.MLParams.test_ratio,
            shuffle=common.MLParams.shuffle_split)
        self.pipe.fit(x_train, y_train, regressor__early_stopping=True)
        y_pred = self.predict(x_test)
        metrics = self.print_metrics(y_test, y_pred, target_name=target_name)
        return metrics, self._score_baselines(baselines, test_mask, y_test)


def binary_scores(y, y_pred):
    if is_binary_target(y):
        return pd.concat([
            describe_vec(y_pred[y == 1], 'positives'),
            describe_vec(y_pred[y == 0], 'negatives')],
            sort=False)


def describe_vec(vec, name):
    return pd.Series(vec).describe().to_frame(name).transpose()


def is_binary_target(y):
    return sorted(list(set(y))) == [0, 1]


def score_metrics(y_true, y_pred):
    metrics = {
        R2: r2_score(y_true, y_pred),
        SPEARMAN: scipy.stats.spearmanr(y_true, y_pred)[0]}
    if is_binary_target(y_true):
        metrics['auc'] = roc_auc_score(y_true, y_pred)
        metrics['apr'] = average_precision_score(y_true, y_pred)
    return metrics


def all_subsets(arr):
    return itertools.chain(*map(
        lambda i: itertools.combinations(arr, i), range(1, len(arr) + 1)))
