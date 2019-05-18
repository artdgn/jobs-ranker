import itertools

import pandas as pd
import numpy as np
import scipy.stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from utils.logger import logger
import common


class RegressorTrainer:

    def __init__(self, test_ratio=common.MLParams.test_ratio, target_name=''):
        # self.cat_cols = cat_cols
        # self.num_cols = num_cols
        # self.y_col = y_col
        # self.eval_on_test = eval_on_test
        self.target_name = target_name
        self.test_ratio = test_ratio

    def train_regressor(self, df, cat_cols, num_cols, y_col, select_cols=False):

        if df[y_col].isnull().sum():
            raise ValueError('Target column contains nans')

        x, y = df[cat_cols + num_cols], df[y_col].values

        rf_pipe = RFPipeline(cat_cols, num_cols)

        metric = 'apr' if is_binary_target(y) else 'r2'

        if select_cols:
            rf_pipe = self.exhaustive_column_selection(
                cat_cols=cat_cols, num_cols=num_cols, x=x, y=y, metric=metric)

        # refit
        rf_pipe.pipe.fit(x, y)

        metrics = rf_pipe.score_and_print_reg_report(
            y, target_name=self.target_name)

        rf_pipe.print_top_n_features(
            x, y, n=common.InfoParams.top_n_feat, target_name=self.target_name)

        model_score = metrics[metric]

        return rf_pipe.pipe, model_score

    def exhaustive_column_selection(self, cat_cols, num_cols, x, y, metric):
        res = []

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_ratio)

        for cols in all_subsets(cat_cols + num_cols):
            rf_pipe = RFPipeline(
                [col for col in cat_cols if col in cols],
                [col for col in num_cols if col in cols])

            test_metrics = rf_pipe.score_regressor_on_test(
                x_train[list(cols)], x_test[list(cols)],
                y_train, y_test)

            res.append((test_metrics[metric], cols))

            logger.info(
                f'selection {test_metrics} {(test_metrics[metric], cols)}')

        best_cols = sorted(res)[-1][1]
        logger.info(f'best: {best_cols}')

        return RFPipeline(
            [col for col in cat_cols if col in best_cols],
            [col for col in num_cols if col in best_cols])


def all_subsets(arr):
    return itertools.chain(*map(
        lambda i: itertools.combinations(arr, i), range(1, len(arr) + 1)))


class FunctionTransformerFeatNames(FunctionTransformer):

    def __init__(self, func=None, name='', validate=False):
        super().__init__(func=func, validate=validate)
        self._name = name

    def get_feature_names(self):
        return [self._name]


class PipelineFeatNames(Pipeline):

    def get_feature_names(self):
        return [n for n in self.steps[-1][1].get_feature_names()]


class RFPipeline:

    def __init__(self, cat_cols, num_cols):
        self.rf = RandomForestRegressor(
            n_estimators=common.MLParams.rf_n_estimators, oob_score=True, n_jobs=-1)

        self.transformer = FeatureUnion([
            *(('tfidf_' + col, self._tfidf_pipe(col)) for col in cat_cols),
            *(('noop_' + col, self._noop_pipe(col)) for col in num_cols),
        ])

        self.pipe = PipelineFeatNames([
            ('transformer', self.transformer),
            ('regressor', self.rf)])

    @staticmethod
    def _tfidf_pipe(col):
        return PipelineFeatNames([
            ('extract_docs', FunctionTransformerFeatNames(
                lambda x: x[col].values, name=col, validate=False)),
            ('tfidf_' + col, TfidfVectorizer(
                ngram_range=common.MLParams.rf_tfidf_ngram_range,
                min_df=common.MLParams.rf_tfidf_min_df,
                stop_words='english'))])

    @staticmethod
    def _noop_pipe(col):
        return PipelineFeatNames([
            ('noop_' + col, FunctionTransformerFeatNames(
                lambda x: x[col].values.reshape(-1, 1),
                name=col,
                validate=False))])

    def print_top_n_features(self, x, y, n=30, target_name=''):
        # names
        top_n_feat = np.argsort(self.rf.feature_importances_)[-n:]
        feat_names = self.transformer.get_feature_names()
        top_names = np.array(feat_names)[top_n_feat]

        # correlations
        x = self.transformer.transform(x)
        top_feat_x = x[:, top_n_feat].toarray()
        cors_mat, _ = scipy.stats.spearmanr(top_feat_x, y.reshape(-1, 1))
        cors_vec = cors_mat[-1, 0:-1]
        df = pd.DataFrame(
            {'name': top_names, 'correlation': cors_vec}). \
            sort_values('correlation', ascending=False)

        logger.info(f'Top {n} informative features and correlations to '
                    f'{target_name}: \n{df}')

    def score_regressor_on_test(self, x_train, x_test, y_train, y_test):
        self.pipe.fit(x_train, y_train)
        y_pred = self.pipe.predict(x_test)
        return score_metrics(y_test, y_pred)

    @staticmethod
    def describe_vec(vec, name):
        return pd.Series(vec).describe().to_frame(name).transpose()

    def score_and_print_reg_report(self, y, target_name=''):
        y_pred = self.rf.oob_prediction_
        metrics = score_metrics(y, y_pred)
        if is_binary_target(y):
            oob_scores = pd.concat([
                self.describe_vec(self.rf.oob_prediction_[y == 1], 'positives'),
                self.describe_vec(self.rf.oob_prediction_[y == 0], 'negatives')])
            logger.info(f"{target_name}, oob scores:\n {oob_scores}")
        logger.info(
            f"\n {pd.Series(metrics).to_frame(f'{target_name} :').transpose()}")
        return metrics


def is_binary_target(y):
    return sorted(list(set(y))) == [0, 1]


def score_metrics(y_true, y_pred):
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'spearman': scipy.stats.spearmanr(y_true, y_pred)[0]}
    if is_binary_target(y_true):
        metrics['auc'] = roc_auc_score(y_true, y_pred)
        metrics['apr'] = average_precision_score(y_true, y_pred)
    return metrics
