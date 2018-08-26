import json
import os

import pandas as pd
import re

from joblist.labeled_jobs import LabeledJobs


class JobsListLabeler:

    keyword_score_col = 'keyword_score'
    model_score_col = 'model_score'

    def __init__(self, scraped, keywords, labeled, older_scraped=()):
        self.scraped_source = scraped
        self.keywords_source = keywords
        self.labeled_source = labeled
        self.older_scraped = older_scraped
        self.regressor = None
        self.model_score = None
        self._pandas_console_options()
        self._load_and_process_data()

    def _load_and_process_data(self):
        self._read_scraped()
        self._read_keywords()
        self._read_labeled()
        self._process_df()

    def _read_keywords(self):
        keywords = self.keywords_source
        if isinstance(keywords, dict):
          self.keywords = keywords
        elif isinstance(keywords, str):
            if os.path.exists(keywords):
                with open(keywords, 'rt') as f:
                    self.keywords = json.load(f)
            else:
                self.keywords = json.loads(keywords)

    def _read_labeled(self):
        self.labels_store = LabeledJobs(self.labeled_source)

    def _read_scrapy_file(self, filename):
        df = pd.read_csv(filename)
        drop_cols = [col for col in df.columns if col.startswith('download_')] + \
                    ['depth']
        df.drop(drop_cols, axis=1, inplace=True)
        return df

    def _read_scraped(self):
        self.df_jobs = self._read_scrapy_file(self.scraped_source)

    def label_jobs(self):
        labeling = True
        prompt = 'y/n/label/stop/refresh?'
        while labeling:
            for ind, row in self.df_jobs.drop('description', axis=1).iterrows():
                if not self.labels_store.labeled(row.url):
                    resp = input(str(row) + '\n' + prompt)
                    if resp == 'stop':
                        labeling = False
                        break
                    if resp == 'refresh':
                        self._load_and_process_data()
                        break
                    self.labels_store.label(row.url, resp)

    @staticmethod
    def _pandas_console_options():
        pd.set_option('display.max_colwidth', 300)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

    @staticmethod
    def _extract_numeric_fields_on_row(row):
        row['description'] = str(row['description']).lower().replace('\n', ' ').replace('\t', ' ')

        # salary
        sal_str = str(row['salary'])
        sal_nums = re.findall('[0-9]+', sal_str.replace(',', ''))
        sal_mult = ('year' in sal_str) * 1 + ('day' in sal_str) * 200 + ('hour' in sal_str) * 1600
        if len(sal_nums) == 2:
            row['salary_low'] = float(sal_nums[0]) * sal_mult
            row['salary_high'] = float(sal_nums[1]) * sal_mult

        # date
        date_str = str(row['date'])
        date_nums = re.findall('[0-9]+', date_str)
        date_mult = ('day' in date_str) * 1 + ('month' in date_str) * 30 + ('hour' in date_str) * 0.04
        if len(date_nums) == 1:
            row['days_age'] = int(int(date_nums[0]) * date_mult)
        return row

    def _extract_numeric_fields(self):
        self.df_jobs = self.df_jobs.apply(self._extract_numeric_fields_on_row, axis=1)
        self.df_jobs.drop(['salary', 'date'], axis=1, inplace=True)

    def _process_df(self):
        self._extract_numeric_fields()
        self._add_keyword_score()
        self._add_model_score()
        self._sort_jobs()

    def _sort_jobs(self):
        sort_col = self.keyword_score_col
        if self.model_score is not None and self.model_score > 0.2:
            sort_col = self.model_score_col
        self.df_jobs.sort_values(sort_col, ascending=False, inplace=True)

    def _add_keyword_score(self):

        def named_regex(word_list, name):
            return '(?P<%s>' % name + '|'.join(word_list) + ')'

        def keyword_density_func(keyword_list, group_name):
            regex = named_regex(keyword_list, group_name)
            return lambda s: len(re.findall(regex, s.lower())) / len(s)

        for kind in ['desc_neg', 'desc_pos']:
            self.df_jobs[kind + '_count'] = \
                self.df_jobs.description.apply(keyword_density_func(self.keywords[kind], kind))

        for kind in ['title_neg', 'title_pos']:
            self.df_jobs[kind + '_count'] = \
                self.df_jobs.title.apply(keyword_density_func(self.keywords[kind], kind))

        self.df_jobs[self.keyword_score_col] = \
            1 / self.df_jobs.desc_pos_count.rank(ascending=False) - \
            1 / self.df_jobs.desc_neg_count.rank(ascending=False) + \
            1 / self.df_jobs.title_pos_count.rank(ascending=False) - \
            1 / self.df_jobs.title_neg_count.rank(ascending=False)

    def _add_model_score(self):
        files = [self.scraped_source] + list(self.older_scraped)

        df_jobs_all = pd.concat(
            [self._read_scrapy_file(file) for file in files], axis=0).\
            drop_duplicates()

        df_join = self.labels_store.df.set_index('url').\
            join(df_jobs_all.set_index('url'), how='left')

        df_join['target'] = df_join['label'].str.contains('y').astype(int)

        feature_col = 'description'

        df_join.dropna(subset=[feature_col], inplace=True)

        self.regressor, self.model_score = reg_test(df_join, x_col=feature_col, y_col='target')

        self.df_jobs[self.model_score_col] = self.regressor.predict(self.df_jobs['description'])


def reg_test(df, x_col, y_col):

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    x, y = df[x_col].values, df[y_col].values

    reg = RandomForestRegressor(n_estimators=50, max_features="sqrt", oob_score=True, n_jobs=1)

    if isinstance(x.ravel()[0], str):
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,3), min_df=3, stop_words='english')),
            ('regressor', reg)])

    elif len(x.shape) == 1:
        pipe = reg
        x = x.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # eval
    pipe.fit(x_train, y_train)

    # score
    r2_test = r2_score(y_test, pipe.predict(x_test))
    # oob_score = r2_score(y_train, reg.oob_prediction_)
    print('r2 test:', r2_test)
    # print('r2 oob:', oob_score)
    print('oob score train:', reg.oob_score_)

    # refit
    pipe.fit(x, y)
    print('oob score all:', reg.oob_score_)
    model_score = reg.oob_score_

    # predict
    # df[result_col] = reg.predict(x)
    return pipe, model_score

