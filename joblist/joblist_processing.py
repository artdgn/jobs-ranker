import json
import os
from itertools import product

import pandas as pd
import pandas.errors
import re

from joblist.labeled_jobs import LabeledJobs
from modeling.descriptions_similarity import dedup_by_descriptions_similarity
from modeling.regression import RegTrainer
from tasks.config import get_task_config, TaskConfig

from utils.logger import logger


class JobsListLabeler:

    keyword_score_col = 'keyword_score'
    model_score_col = 'model_score'
    salary_guess_col = 'salary_guess'

    def __init__(self, scraped, task_config: TaskConfig,
                 older_scraped=(), dedup_new=True,
                 dup_keep='first', skipped_as_negatives=True):
        self.scraped_source = scraped
        self.task_config = task_config
        self.older_scraped = older_scraped
        self.dedup_new = dedup_new
        self.dup_keep = dup_keep
        self.skipped_as_negatives = skipped_as_negatives
        self.regressor = None
        self.model_score = None
        self.regressor_salary = None
        self.reg_sal_model_score = None
        self.intermidiate_score_cols = []
        self.dup_dict = {}
        self._pandas_console_options()
        self._load_and_process_data()


    def _load_and_process_data(self):
        self._read_all_scraped()
        self._read_labeled()
        self._read_last_scraped(dedup=self.dedup_new)
        if len(self.df_jobs):
            self._read_task_config()
            self._process_df()


    def _recalc(self):
        self._read_task_config()
        self._process_df()


    def _read_task_config(self):
        self.task_config = get_task_config(self.task_config.name)


    def _read_labeled(self):
        self.labels_dao = LabeledJobs(task_name=self.task_config.name,
                                      dup_dict = self.dup_dict)
        logger.info(f'total labeled jobs DF: {len(self.labels_dao.df)}')


    def _read_scrapy_file(self, filename):
        try:
            df = pd.read_csv(filename)
        except pandas.errors.EmptyDataError as e:
            logger.info(f'found empty scrape file:{filename}. trying to delete.')
            os.remove(filename)
            return pd.DataFrame()
        else:
            drop_cols = [col for col in df.columns if col.startswith('download_')] + \
                        ['depth']
            df.drop(drop_cols, axis=1, inplace=True)
            df['scraped_file'] = filename
            return df


    def _read_last_scraped(self, dedup=True):
        if not dedup:
            self.df_jobs = self._read_scrapy_file(self.scraped_source)
            self._add_duplicates_column()
        else:
            self.df_jobs = self.df_jobs_all.\
                               loc[self.df_jobs_all['scraped_file'] == self.scraped_source, :]
        logger.info(f'most recent scrape DF: {len(self.df_jobs)} ({self.scraped_source}, '
                    f'all scraped: {len(self._read_scrapy_file(self.scraped_source))})')


    def _add_duplicates_column(self):
        dup_no_self = {k: [u for u in v if u != k] for k, v in self.dup_dict.items()}
        df_dups = pd.DataFrame({'url': list(dup_no_self.keys()),
                                'duplicates': list(dup_no_self.values())})
        self.df_jobs = pd.merge(self.df_jobs, df_dups, on='url', how='left')


    def _read_all_scraped(self):
        files = list(self.older_scraped) + [self.scraped_source]

        df_jobs = pd.concat(
            [self._read_scrapy_file(file) for file in files], axis=0). \
            drop_duplicates(subset=['url']).\
            dropna(subset=['description'])

        keep_inds, dup_dict_inds = dedup_by_descriptions_similarity(
            df_jobs['description'], keep=self.dup_keep)

        urls = df_jobs['url'].values
        self.dup_dict = {urls[i]: urls[sorted([i] + list(dups))]
                         for i, dups in dup_dict_inds.items()}

        self.df_jobs_all = df_jobs.iloc[keep_inds]

        logger.info(f'total historic jobs DF: {len(self.df_jobs_all)} (deduped from {len(df_jobs)})')


    def label_jobs(self, recalc_everytime=False):

        def get_urls_stack():
            return self.df_jobs['url'].tolist()[::-1]

        prompt = 'y / n / float / stop / recalc / skip ? : '
        not_show_cols = ['description', 'scraped_file', 'salary', 'date'] + \
                        self.intermidiate_score_cols
        urls_stack = get_urls_stack()
        skipped = set()
        while len(urls_stack):
            url = urls_stack.pop()
            if (not self.labels_dao.labeled(url)) and not (url in skipped):
                row = self.df_jobs.loc[self.df_jobs['url']==url].iloc[0]. \
                    drop(not_show_cols).dropna()
                resp = input(str(row) + '\n' + prompt)

                if resp == 'stop':
                    break

                elif resp == 'skip':
                    skipped.add(url)
                    continue

                elif resp == 'recalc':
                    self._recalc()
                    urls_stack = get_urls_stack()
                else:
                    self.labels_dao.label(row.url, resp)
                    if recalc_everytime:
                        self._recalc()
                        urls_stack = get_urls_stack()

        if not len(urls_stack):
            logger.info('No more new unlabeled jobs. Try turning dedup off to go over duplicates.')


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
        # df.drop(['salary', 'date'], axis=1, inplace=True)
        return df


    def _process_df(self):
        self.df_jobs = self._extract_numeric_fields(self.df_jobs)
        self.df_jobs = self._add_keyword_score(self.df_jobs)
        self.df_jobs = self._add_salary_guess(self.df_jobs)
        self.df_jobs = self._add_model_score(self.df_jobs)
        self.df_jobs = self._sort_jobs(self.df_jobs)


    def _sort_jobs(self, df):
        sort_col = [self.keyword_score_col, self.model_score_col]
        if self.model_score is not None and self.model_score > 0.1:
            sort_col.reverse()
        df.sort_values(sort_col, ascending=False, inplace=True)
        return df


    def _add_keyword_score(self, df):

        def named_regex(word_list, name):
            return '(?P<%s>' % name + '|'.join(word_list) + ')'

        def keyword_density_func(keyword_list, group_name):
            regex = named_regex(keyword_list, group_name)
            return lambda s: len(re.findall(regex, s.lower())) / len(s)

        for source, weight in product(['description', 'title'], ['positive', 'negative']):
            kind = '%s_%s' % (source, weight)
            col = kind + '_count'
            df[col] = df[source].apply(keyword_density_func(self.task_config[kind], kind))
            if col not in self.intermidiate_score_cols:
                self.intermidiate_score_cols.append(col)

        df[self.keyword_score_col] = \
            1 / df.description_positive_count.rank(ascending=False) - \
            1 / df.description_negative_count.rank(ascending=False) + \
            1 / df.title_positive_count.rank(ascending=False) - \
            1 / df.title_negative_count.rank(ascending=False)
        return df


    def _add_salary_guess(self, df, refit=False):

        if self.regressor_salary is None or refit:
            self._train_salary_regressor()

        df[self.salary_guess_col] = (self.regressor_salary.predict(df)
                                     if self.regressor_salary else 0)

        return df


    def _train_salary_regressor(self):
        df_train = self.df_jobs_all.copy()
        df_train = self._extract_numeric_fields(df_train)
        df_train = self._add_keyword_score(df_train)

        target_col = 'salary_high'

        # cat_cols = ['description']
        cat_cols = ['description', 'title']
        df_train.dropna(subset=cat_cols + [target_col], inplace=True)

        if len(df_train) > 10:
            num_cols = [self.keyword_score_col]
            self.regressor_salary, self.reg_sal_model_score = \
                RegTrainer(print_prefix='salary: '). \
                    train_regressor(df_train,
                                    cat_cols=cat_cols,
                                    num_cols=num_cols,
                                    y_col=target_col,
                                    select_cols=False)
        else:
            logger.warn(f'Not training salary regressor due to '
                        f'having only {len(df_train)} samples')


    def _add_model_score(self, df, refit=True):

        if self.regressor is None or refit:
            self._train_label_regressor()

        df[self.model_score_col] = (self.regressor.predict(df)
                                    if self.regressor is not None else 0)

        return df


    def _train_label_regressor(self):

        df_jobs_all = self.df_jobs_all.copy()

        labels_df = self.labels_dao.export_df(keep=self.dup_keep)

        df_train = labels_df.set_index('url'). \
            join(df_jobs_all.set_index('url'), how='left')

        df_train['target'] = df_train['label']. \
            replace('y', '1.0'). \
            replace('n', '0.0'). \
            astype(float)

        if self.skipped_as_negatives:
            df_past_skipped = df_jobs_all.loc[
                              ~df_jobs_all.url.isin(
                                  df_train.reset_index()['url'].tolist() +
                                  self.df_jobs['url'].tolist()), :].copy()
            df_past_skipped.loc[:, 'target'] = 0.0
            df_train = df_train.append(df_past_skipped, sort=True)

        # cat_cols = ['description']
        cat_cols = ['description', 'title']

        df_train.dropna(subset=cat_cols, inplace=True)

        if len(df_train) > 10:
            df_train = self._extract_numeric_fields(df_train)
            df_train = self._add_keyword_score(df_train)
            df_train = self._add_salary_guess(df_train)

            num_cols = self.intermidiate_score_cols + [self.keyword_score_col, self.salary_guess_col]
            # num_cols = self.intermidiate_score_cols + [self.keyword_score_col]
            # num_cols = []

            self.regressor, self.model_score = \
                RegTrainer(print_prefix='label: '). \
                    train_regressor(df_train,
                                    cat_cols=cat_cols,
                                    num_cols=num_cols,
                                    y_col='target',
                                    select_cols=False)
        else:
            logger.warn(f'Not training label regressor due to '
                        f'having only {len(df_train)} samples')


