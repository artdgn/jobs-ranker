import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def train_regressor(df, cat_cols, num_cols, y_col, eval_on_test=False, print_prefix=''):

    x, y = df[cat_cols + num_cols], df[y_col].values

    reg = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)

    tfidf = Pipeline([
        ('extract_docs', FunctionTransformer(lambda x: x.values[:,0], validate=False)),
        ('tfidf', TfidfVectorizer(ngram_range=(1,3), min_df=3, stop_words='english'))])

    noop = Pipeline([
        ('nothing', FunctionTransformer(lambda x: x, validate=False))])

    pipe = Pipeline([
        ('transformer', ColumnTransformer(transformers=[
            ('tfidf', tfidf, cat_cols),
            ('noop', noop, num_cols),
        ])),
        ('regressor', reg)])

    if eval_on_test:  # not too important with RF (because of OOB)
        print(print_prefix, 'r2 test:', score_regressor_on_test(pipe, x=x, y=y, ratio=0.3))

    # refit
    pipe.fit(x, y)

    # report
    describe = lambda vec, name: pd.Series(vec).describe().to_frame(name).transpose()
    if 1 in y and 0 in y:
        print(print_prefix, 'oob scores:\n',
              pd.concat([describe(reg.oob_prediction_[y == 1], 'positives'),
                         describe(reg.oob_prediction_[y == 0], 'negatives')]))
    print(print_prefix, 'oob score all:', reg.oob_score_)
    model_score = reg.oob_score_

    return pipe, model_score

def score_regressor_on_test(reg, x, y, ratio=0.3):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio)
    reg.fit(x_train, y_train)
    return r2_score(y_test, reg.predict(x_test))