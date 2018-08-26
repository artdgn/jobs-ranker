import json
import os

import pandas as pd
import re

from joblist.labeled_jobs import LabeledJobs


class JobsListLabeler:

    def __init__(self, scraped, keywords, labeled):
        self.scraped_source = scraped
        self.keywords_source = keywords
        self.labeled_source = labeled
        self._pandas_console_options()
        self._load_and_process_data()

    def _load_and_process_data(self):
        self._read_scraped(self.scraped_source)
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

    def _read_scraped(self, filename):
        self.df_jobs = pd.read_csv(filename)
        drop_cols = [col for col in self.df_jobs.columns if col.startswith('download_')] + \
                    ['depth']
        self.df_jobs.drop(drop_cols, axis=1, inplace=True)
        return self

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
        return self

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

        self.df_jobs['comb_score'] = \
            1 / self.df_jobs.desc_pos_count.rank(ascending=False) - \
            1 / self.df_jobs.desc_neg_count.rank(ascending=False) + \
            1 / self.df_jobs.title_pos_count.rank(ascending=False) - \
            1 / self.df_jobs.title_neg_count.rank(ascending=False)

        self.df_jobs.sort_values('comb_score', ascending=False, inplace=True)

#########
'''
def reg_test(df, x_col, y_col, result_col):

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    df_reg = df[~df[y_col].isnull()][[x_col, y_col]].copy()
    # fit for eval
    train, test = train_test_split(df_reg, test_size=0.3)
    reg = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=None)),
        # ('regressor', Lasso())])
        ('regressor', RandomForestRegressor())])
    reg.fit(train[x_col], train[y_col])
    # score
    print(r2_score(test[y_col], reg.predict(test[x_col])))
    # refit
    reg.fit(df_reg[x_col], df_reg[y_col])
    # predict
    df[result_col] = reg.predict(df[x_col])
    return df

df_guess = reg_test(df_post, x_col='title', y_col='salary_high', result_col='salary_guess')
'''