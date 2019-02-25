import itertools

import pandas as pd
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from utils.logger import logger


class RegTrainer():

    def __init__(self, test_ratio=0.3, print_prefix=''):
        # self.cat_cols = cat_cols
        # self.num_cols = num_cols
        # self.y_col = y_col
        # self.eval_on_test = eval_on_test
        self.print_prefix = print_prefix
        self.test_ratio = test_ratio

    def train_regressor(self, df, cat_cols, num_cols, y_col, select_cols=False):

        if df[y_col].isnull().sum():
            raise ValueError('Target column contains nans')

        x, y = df[cat_cols + num_cols], df[y_col].values

        pipe, reg = build_RF_pipiline(cat_cols, num_cols)

        metric = 'apr' if is_binary_target(y) else 'r2'

        if select_cols:
            pipe, reg = self.exhaustive_column_selection(
                cat_cols=cat_cols, num_cols=num_cols, x=x, y=y, metric=metric)

        # refit
        pipe.fit(x, y)

        metrics = self.score_and_print_report(reg, y)

        model_score = metrics[metric]

        return pipe, model_score

    def score_and_print_report(self, reg, y):
        y_pred = reg.oob_prediction_
        metrics = score_metrics(y, y_pred)
        describe = lambda vec, name: pd.Series(vec).describe().to_frame(name).transpose()
        if is_binary_target(y):
            oob_scores = pd.concat([
                describe(reg.oob_prediction_[y == 1], 'positives'),
                describe(reg.oob_prediction_[y == 0], 'negatives')])
            logger.info(f"{self.print_prefix}, 'oob scores:\n {oob_scores}")
        logger.info(f'\n {pd.Series(metrics).to_frame(self.print_prefix).transpose()}')
        return metrics

    def exhaustive_column_selection(self, cat_cols, num_cols, x, y, metric):
        res = []

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_ratio)

        for cols in all_subsets(cat_cols + num_cols):

            pipe, reg = build_RF_pipiline([col for col in cat_cols if col in cols],
                                          [col for col in num_cols if col in cols])

            test_metrics = score_regressor_on_test(
                pipe, x_train[list(cols)], x_test[list(cols)], y_train, y_test)

            res.append((test_metrics[metric], cols))

            logger.info(f'selection {test_metrics} {(test_metrics[metric], cols)}')

        best_cols = sorted(res)[-1][1]
        logger.info(f'best: {best_cols}')

        pipe, reg = build_RF_pipiline([col for col in cat_cols if col in best_cols],
                                      [col for col in num_cols if col in best_cols])
        return pipe, reg


def all_subsets(arr):
    return itertools.chain(*map(
        lambda i: itertools.combinations(arr, i), range(1, len(arr) + 1)))



def build_RF_pipiline(cat_cols, num_cols):
    reg = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)

    def tfidf(col):
        return Pipeline([
            ('extract_docs', FunctionTransformer(lambda x: x[col].values, validate=False)),
            ('tfidf_' + col, TfidfVectorizer(ngram_range=(1,3), min_df=3, stop_words='english'))])

    def noop(col):
        return Pipeline([
            ('noop_' + col, FunctionTransformer(lambda x: x[col].values.reshape(-1, 1), validate=False))])

    pipe = Pipeline([
        ('transformer', FeatureUnion([
            *(('tfidf_' + col, tfidf(col)) for col in cat_cols),
            *(('noop_' + col, noop(col)) for col in num_cols),
        ])),
        ('regressor', reg)])

    return pipe, reg

def score_regressor_on_test(pipe, x_train, x_test, y_train, y_test):
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    return score_metrics(y_test, y_pred)


def is_binary_target(y):
    return sorted(list(set(y))) == [0, 1]


def score_metrics(y_true, y_pred):
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'spearman': spearmanr(y_true, y_pred)[0]}
    if is_binary_target(y_true):
        metrics['auc'] = roc_auc_score(y_true, y_pred)
        metrics['apr'] = average_precision_score(y_true, y_pred)
    return metrics
