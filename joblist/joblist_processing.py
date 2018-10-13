import json
import os
from itertools import product

import pandas as pd
import re

from joblist.labeled_jobs import LabeledJobs
from modeling.edit_distance_similarity import dedup_by_descriptions_similarity
from modeling.regression import RegTrainer


class JobsListLabeler:

    keyword_score_col = 'keyword_score'
    model_score_col = 'model_score'
    salary_guess_col = 'salary_guess'

    def __init__(self, scraped, keywords, labeled, older_scraped=(), dedup_new=True, dup_keep='first'):
        self.scraped_source = scraped
        self.keywords_source = keywords
        self.labeled_source = labeled
        self.older_scraped = older_scraped
        self.dedup_last = dedup_new
        self.dup_keep = dup_keep
        self.regressor = None
        self.model_score = None
        self.regressor_salary = None
        self.reg_sal_model_score = None
        self.intermidiate_score_cols = []
        self.dup_dict = {}
        self._pandas_console_options()
        self._load_and_process_data()

    def _load_and_process_data(self):
        self._read_labeled()
        self._read_all_scraped()
        self._read_last_scraped(dedup=self.dedup_last)
        self._read_keywords()
        self._process_df()

    def _recalc(self):
        self._read_keywords()
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
        self.labels_dao = LabeledJobs(self.labeled_source)

    def _read_scrapy_file(self, filename):
        df = pd.read_csv(filename)
        drop_cols = [col for col in df.columns if col.startswith('download_')] + \
                    ['depth']
        df.drop(drop_cols, axis=1, inplace=True)
        df['scraped_file'] = filename
        return df

    def _read_last_scraped(self, dedup=True):
        if not dedup:
            self.df_jobs = self._read_scrapy_file(self.scraped_source)
        else:
            self.df_jobs = self.df_jobs_all.loc[self.df_jobs_all['scraped_file'] ==
                                                self.scraped_source, :]

    def _read_all_scraped(self):
        files = list(self.older_scraped) + [self.scraped_source]

        df_jobs = pd.concat(
            [self._read_scrapy_file(file) for file in files], axis=0). \
            drop_duplicates(subset=['url']).\
            dropna(subset=['description'])

        keep_inds, self.dup_dict = dedup_by_descriptions_similarity(
            df_jobs['description'], keep=self.dup_keep)

        self.labels_dao.dedup(self.dup_dict, df_jobs['url'].values, keep=self.dup_keep)

        self.df_jobs_all = df_jobs.iloc[keep_inds]

    def label_jobs(self, recalc_everytime=True):
        labeling = True
        prompt = 'y/n/float/stop/recalc?'
        while labeling:
            for ind, row in self.df_jobs.drop(['description', 'scraped_file'], axis=1).iterrows():
                if not self.labels_dao.labeled(row.url):
                    row = row.drop(self.intermidiate_score_cols).dropna()
                    resp = input(str(row) + '\n' + prompt)
                    if resp == 'stop':
                        labeling = False
                        break
                    if resp == 'recalc':
                        self._recalc()
                        break
                    self.labels_dao.label(row.url, resp)
                    if recalc_everytime:
                        self._recalc()

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

    def _extract_numeric_fields(self, df):
        df = df.apply(self._extract_numeric_fields_on_row, axis=1)
        df.drop(['salary', 'date'], axis=1, inplace=True)
        return df

    def _process_df(self):
        self.df_jobs = self._extract_numeric_fields(self.df_jobs)
        self.df_jobs = self._add_keyword_score(self.df_jobs)
        self.df_jobs = self._add_salary_guess(self.df_jobs)
        self.df_jobs = self._add_model_score(self.df_jobs)
        self.df_jobs = self._sort_jobs(self.df_jobs)

    def _sort_jobs(self, df):
        sort_col = self.keyword_score_col
        if self.model_score is not None and self.model_score > 0.2:
            sort_col = self.model_score_col
        df.sort_values(sort_col, ascending=False, inplace=True)
        return df

    def _add_keyword_score(self, df):

        def named_regex(word_list, name):
            return '(?P<%s>' % name + '|'.join(word_list) + ')'

        def keyword_density_func(keyword_list, group_name):
            regex = named_regex(keyword_list, group_name)
            return lambda s: len(re.findall(regex, s.lower())) / len(s)

        for source, weight in product(['description', 'title'], ['pos', 'neg']):
            kind = '%s_%s' % (source, weight)
            col = kind + '_count'
            df[col] = df[source].apply(keyword_density_func(self.keywords[kind], kind))
            if col not in self.intermidiate_score_cols:
                self.intermidiate_score_cols.append(col)

        df[self.keyword_score_col] = \
            1 / df.description_pos_count.rank(ascending=False) - \
            1 / df.description_neg_count.rank(ascending=False) + \
            1 / df.title_pos_count.rank(ascending=False) - \
            1 / df.title_neg_count.rank(ascending=False)
        return df

    def _add_salary_guess(self, df, refit=False):

        if self.regressor_salary is None or refit:

            df_jobs = self.df_jobs_all.copy()
            df_jobs = self._extract_numeric_fields(df_jobs)
            df_jobs = self._add_keyword_score(df_jobs)

            target_col = 'salary_high'

            # cat_cols = ['description']
            cat_cols = ['description', 'title']
            df_jobs.dropna(subset=cat_cols + [target_col], inplace=True)
            num_cols = [self.keyword_score_col]
            self.regressor_salary, self.reg_sal_model_score = \
                RegTrainer(print_prefix='salary: ').\
                    train_regressor(df_jobs,
                                    cat_cols=cat_cols,
                                    num_cols=num_cols,
                                    y_col=target_col,
                                    select_cols=False)

        df[self.salary_guess_col] = self.regressor_salary.predict(df)

        return df

    def _add_model_score(self, df, refit=True, skipped_as_negatives=True):

        if self.regressor is None or refit:

            df_jobs_all = self.df_jobs_all.copy()

            df_join = self.labels_dao.df.set_index('url').\
                join(df_jobs_all.set_index('url'), how='left')

            df_join['target'] = df_join['label'].\
                replace('y', '1.0').\
                replace('n', '0.0').\
                astype(float)

            if skipped_as_negatives:
                df_past_skipped = df_jobs_all.loc[
                                  ~df_jobs_all.url.isin(
                                      df_join.reset_index()['url'].tolist() +
                                      self.df_jobs['url'].tolist()), :].copy()
                df_past_skipped.loc[:, 'target'] = 0.0
                df_join = df_join.append(df_past_skipped, sort=True)

            # cat_cols = ['description']
            cat_cols = ['description', 'title']

            df_join.dropna(subset=cat_cols, inplace=True)

            df_join = self._extract_numeric_fields(df_join)
            df_join = self._add_keyword_score(df_join)
            df_join = self._add_salary_guess(df_join)

            num_cols = self.intermidiate_score_cols + [self.keyword_score_col, self.salary_guess_col]
            # num_cols = self.intermidiate_score_cols + [self.keyword_score_col]
            # num_cols = []

            self.regressor, self.model_score = \
                RegTrainer(print_prefix='label: ').\
                    train_regressor(df_join,
                                    cat_cols=cat_cols,
                                    num_cols=num_cols,
                                    y_col='target',
                                    select_cols=False)

        df[self.model_score_col] = self.regressor.predict(df)

        return df



