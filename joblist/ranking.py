from itertools import product

import pandas as pd
import re

import common
from crawler.scraping import read_scrapy_file
from joblist.labeled import LabeledJobs
from ml.descriptions_similarity import dedup_by_descriptions_similarity
from ml.regression import RegressorTrainer
from tasks.config import get_task_config, TaskConfig

from utils.logger import logger


class JobsRanker:
    keyword_score_col = 'keyword_score'
    model_score_col = 'model_score'
    salary_guess_col = 'salary_guess'
    pos_label = 'y'
    neg_label = 'n'

    def __init__(self, scraped, task_config: TaskConfig,
                 older_scraped=(),
                 dedup_new=True,
                 dup_keep=common.MLParams.dedup_keep,
                 skipped_as_negatives=True
                 ):
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
        self._load_and_process_data()
        _pandas_console_options()

    def _load_and_process_data(self):
        self._read_all_scraped()
        self._init_labeled_dao()
        self._read_last_scraped(dedup=self.dedup_new)
        if len(self.df_jobs):
            self.rank_jobs()

    def rank_jobs(self):
        self._read_task_config()
        self.df_jobs = self._add_model_score(self.df_jobs)
        self.df_jobs = self._sort_jobs(self.df_jobs)

    def _read_task_config(self):
        self.task_config = get_task_config(self.task_config.name)

    def _init_labeled_dao(self):
        self.labels_dao = LabeledJobs(task_name=self.task_config.name,
                                      dup_dict=self.dup_dict)
        logger.info(f'total labeled jobs DF: {len(self.labels_dao.df)}')

    def _read_last_scraped(self, dedup=True):
        if not dedup:
            self.df_jobs = read_scrapy_file(self.scraped_source)
            self._add_duplicates_column()
        else:
            self.df_jobs = self.df_jobs_all. \
                               loc[self.df_jobs_all['scraped_file'] == self.scraped_source, :]
        logger.info(f'most recent scrape DF: {len(self.df_jobs)} ({self.scraped_source}, '
                    f'all scraped: {len(read_scrapy_file(self.scraped_source))})')

    def _add_duplicates_column(self):
        dup_no_self = {k: [u for u in v if u != k]
                       for k, v in self.dup_dict.items()}
        dups_lists = [(l if l else None) for l in list(dup_no_self.values())]
        df_dups = pd.DataFrame({
            'url': list(dup_no_self.keys()),
            'duplicates': dups_lists,
            # 'duplicates_avg_label':
        })
        self.df_jobs = pd.merge(self.df_jobs, df_dups, on='url', how='left')

    def _read_all_scraped(self):
        files = list(self.older_scraped) + [self.scraped_source]

        df_jobs = pd.concat(
            [read_scrapy_file(file) for file in files], axis=0). \
            drop_duplicates(subset=['url']). \
            dropna(subset=['description'])

        keep_inds, dup_dict_inds = dedup_by_descriptions_similarity(
            df_jobs['description'], keep=self.dup_keep)

        urls = df_jobs['url'].values
        self.dup_dict = {urls[i]: urls[sorted([i] + list(dups))]
                         for i, dups in dup_dict_inds.items()}

        self.df_jobs_all = df_jobs.iloc[keep_inds]

        logger.info(f'total historic jobs DF: {len(self.df_jobs_all)} '
                    f'(deduped from {len(df_jobs)})')

    def displayable_job_by_url(self, url):
        not_show_cols = (['description', 'scraped_file', 'salary', 'date'] +
                         self.intermidiate_score_cols)
        row = self.df_jobs.loc[self.df_jobs['url'] == url].iloc[0]
        return row.drop(not_show_cols).dropna()

    def get_sorted_urls_stack(self):
        return self.df_jobs['url'].tolist()[::-1]

    def is_valid_label_input(self, label: str):
        try:
            number = float(label.
                           replace(self.pos_label, '1.0').
                           replace(self.neg_label, '0.0'))
            if not 0 <= number <= 1:
                raise ValueError

        except ValueError:
            logger.error(f'Invalid input : {label}')
            return False

    @staticmethod
    def _extract_numeric_fields(df):
        if not all(col in df.columns
                   for col in ['days_age', 'salary_low', 'salary_high']):
            df = df.apply(_extract_numeric_fields_on_row, axis=1)
        return df

    def _sort_jobs(self, df):
        sort_cols = [self.model_score_col, self.keyword_score_col]
        if self.model_score is None or \
                self.model_score < common.MLParams.min_model_score:
            sort_cols.reverse()
            logger.info(f'Sorting by keyword-score instead of model-score '
                        f'because model-score = {self.model_score}')
        logger.info(f'Sorting by columns: {sort_cols}')
        df.sort_values(sort_cols, ascending=False, inplace=True)
        return df

    def _add_keyword_features(self, df):

        def group_named_regex(word_list, name):
            return '(?P<%s>' % name + '|'.join(word_list) + ')'

        for source, weight in product(['description', 'title'],
                                      ['positive', 'negative']):

            group_kind = f'{source}_{weight}'
            group_col = group_kind + '_count'
            keywords = self.task_config[group_kind]
            group_regex = group_named_regex(keywords, group_kind)

            df[group_col] = df[source].str.count(group_regex)

            if group_col not in self.intermidiate_score_cols:
                self.intermidiate_score_cols.append(group_col)

        df[self.keyword_score_col] = (
            1 / df.description_positive_count.rank(ascending=False)
            - 1 / df.description_negative_count.rank(ascending=False)
            + 1 / df.title_positive_count.rank(ascending=False)
            - 1 / df.title_negative_count.rank(ascending=False))

        return df

    def _add_salary_guess(self, df, refit=False):
        df = self._add_salary_features(df)

        if self.regressor_salary is None or refit:
            self._train_salary_regressor()

        df[self.salary_guess_col] = (self.regressor_salary.predict(df)
                                     if self.regressor_salary else 0)

        return df

    def _add_salary_features(self, df):
        df = self._extract_numeric_fields(df)
        df = self._add_keyword_features(df)
        return df

    def _train_salary_regressor(self):
        df_train = self.df_jobs_all.copy()
        df_train = self._add_salary_features(df_train)

        target_col = 'salary_high'

        # cat_cols = ['description']
        cat_cols = ['description', 'title']
        df_train.dropna(subset=cat_cols + [target_col], inplace=True)

        if len(df_train) >= common.MLParams.min_training_samples:
            num_cols = [self.keyword_score_col]
            trainer = RegressorTrainer(target_name='salary')
            self.regressor_salary, self.reg_sal_model_score = (
                trainer.train_regressor(df_train,
                                        cat_cols=cat_cols,
                                        num_cols=num_cols,
                                        y_col=target_col,
                                        select_cols=False))
        else:
            logger.warn(f'Not training salary regressor due to '
                        f'having only {len(df_train)} samples')

    def _add_model_score(self, df, refit=True):
        df = self._add_relevance_features(df)

        if self.regressor is None or refit:
            self._train_label_regressor()

        df[self.model_score_col] = (self.regressor.predict(df)
                                    if self.regressor is not None else 0)
        return df

    def _add_relevance_features(self, df):
        df = self._extract_numeric_fields(df)
        df = self._add_keyword_features(df)
        df = self._add_salary_guess(df)
        return df

    def _train_df_with_labels(self):
        df_jobs_all = self.df_jobs_all.copy()

        labels_df = self.labels_dao.export_df(keep=self.dup_keep)

        df_train = labels_df.set_index('url'). \
            join(df_jobs_all.set_index('url'), how='left')

        df_train['target'] = df_train['label']. \
            replace(self.pos_label, '1.0'). \
            replace(self.neg_label, '0.0'). \
            astype(float)

        if self.skipped_as_negatives:
            df_past_skipped = df_jobs_all.loc[
                              ~df_jobs_all.url.isin(
                                  df_train.reset_index()['url'].tolist() +
                                  self.df_jobs['url'].tolist()), :].copy()
            df_past_skipped.loc[:, 'target'] = 0.0
            df_train = df_train.append(df_past_skipped, sort=True)

        return df_train

    def _train_label_regressor(self):

        df_train = self._train_df_with_labels()

        cat_cols = ['description', 'title']

        df_train.dropna(subset=cat_cols, inplace=True)

        if len(df_train) >= common.MLParams.min_training_samples:
            df_train = self._add_relevance_features(df_train)

            num_cols = (self.intermidiate_score_cols +
                        [self.keyword_score_col, self.salary_guess_col])

            trainer = RegressorTrainer(target_name='label')
            self.regressor, self.model_score = (
                trainer.train_regressor(df_train,
                                        cat_cols=cat_cols,
                                        num_cols=num_cols,
                                        y_col='target',
                                        select_cols=False))

        else:
            logger.warn(f'Not training label regressor due to '
                        f'having only {len(df_train)} samples')

    def is_labeled(self, url):
        return self.labels_dao.labeled(url)

    def add_label(self, url, label):
        if self.is_valid_label_input(label):
            return self.labels_dao.label(url, label)
        else:
            raise ValueError()


def _extract_numeric_fields_on_row(row):
    row['description'] = (
        str(row['description']).lower().
            replace('\n', ' ').replace('\t', ' '))

    row['description_length'] = len(row['description'])

    # salary
    sal_str = str(row['salary'])
    sal_nums = re.findall('[0-9]+', sal_str.replace(',', ''))
    sal_mult = (('year' in sal_str) * 1
                + ('day' in sal_str) * 200
                + ('hour' in sal_str) * 1600)
    if len(sal_nums) == 2:
        row['salary_low'] = float(sal_nums[0]) * sal_mult
        row['salary_high'] = float(sal_nums[1]) * sal_mult

    # date
    date_str = str(row['date'])
    date_nums = re.findall('[0-9]+', date_str)
    date_mult = (('day' in date_str) * 1
                 + ('month' in date_str) * 30
                 + ('hour' in date_str) * 0.04)
    if len(date_nums) == 1:
        row['days_age'] = int(int(date_nums[0]) * date_mult)
    return row


def _pandas_console_options():
    pd.set_option('display.max_colwidth', 300)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
