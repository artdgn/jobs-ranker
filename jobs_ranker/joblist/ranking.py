import abc
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from threading import Lock

import numpy
import pandas as pd

from jobs_ranker.config import common
from jobs_ranker.joblist.labeled import LabeledJobs, LabelsAPI
from jobs_ranker.ml import regression
from jobs_ranker.ml.deduplication import calc_duplicates
from jobs_ranker.scraping.crawling import CrawlsFilesDao
from jobs_ranker.tasks.configs import TaskConfig, TasksConfigsDao
from jobs_ranker.utils.instrumentation import LogCallsTimeAndOutput
from jobs_ranker.utils.logger import logger


class RankerAPI(abc.ABC):

    @abc.abstractmethod
    def __init__(self,
                 task_config: TaskConfig,
                 dedup_recent=True
                 ):
        self.task_config = task_config
        self.dedup_recent = dedup_recent

        self.sort_col = None
        self.recent_crawl_source = None

    @property
    @abc.abstractmethod
    def loaded(self):
        return False

    @property
    @abc.abstractmethod
    def busy(self):
        return False

    @property
    @abc.abstractmethod
    def labeler(self):
        return LabelsAPI()

    @abc.abstractmethod
    def load_and_process_data(self, background=False):
        pass

    @abc.abstractmethod
    def url_data(self, url):
        return pd.Series()

    @abc.abstractmethod
    def next_unlabeled(self):
        return ''

    @abc.abstractmethod
    def rerank_jobs(self, background=False):
        pass

    @property
    @abc.abstractmethod
    def ranking_scores(self):
        return ''


def get_ranker(task_config: TaskConfig,
               dedup_new=True) -> RankerAPI:
    return JobsRanker(task_config=task_config,
                      dedup_recent=dedup_new)


class JobsRanker(RankerAPI, LogCallsTimeAndOutput):
    keyword_score_col = 'keyword_score'
    model_score_col = 'model_score'
    salary_guess_col = 'salary_guess'
    years_experience_col = 'years_exp_max'
    scrape_order_rank_col = 'scrape_order_rank'
    target_col = 'target'
    description_col = 'description'
    title_col = 'title'
    text_cols = [description_col, title_col]

    def __init__(self,
                 task_config: TaskConfig,
                 dedup_recent=True):
        super().__init__(task_config=task_config,
                         dedup_recent=dedup_recent)
        self.regressor = None
        self.model_score = None
        self.keyword_score = None
        self.scrape_order_score = None
        self.regressor_salary = None
        self.reg_sal_model_score = None
        self.df_recent = None
        self.df_all_deduped = None
        self.df_all_read = None
        self.intermidiate_score_cols = []
        self._labels_dao = None

        self.dup_dict = {}
        self._unlabeled = None

        self._bg_executor = ThreadPoolExecutor(max_workers=1)
        self._busy_lock = Lock()
        self._bg_future = None

    @property
    def num_cols_salary(self):
        return [self.keyword_score_col,
                self.years_experience_col,
                self.scrape_order_rank_col]

    @property
    def num_cols_label(self):
        return (self.intermidiate_score_cols +
                [self.keyword_score_col,
                 self.salary_guess_col,
                 self.years_experience_col,
                 self.scrape_order_rank_col])

    @property
    def loaded(self):
        self._check_bg_thread()
        return self.df_recent is not None

    @property
    def busy(self):
        return self._busy_lock.locked()

    def _check_bg_thread(self):
        if self._bg_future is not None and not self._bg_future.running():
            self._bg_future.result()

    @property
    def labeler(self):
        if self._labels_dao is None:
            if not self.dup_dict:
                raise ValueError(f'dup_dict is not set')
            self._labels_dao = LabeledJobs(task_name=self.task_config.name,
                                           dup_dict=self.dup_dict)
        return self._labels_dao

    def _do_in_background(self, func):
        while self._bg_future is not None and self._bg_future.running():
            return self._bg_future.result()
        self._bg_future = self._bg_executor.submit(func)
        # check for errors
        time.sleep(0.1)
        self._check_bg_thread()

    def load_and_process_data(self, background=False):
        if background:
            self._do_in_background(self.load_and_process_data)
        else:
            self._load_and_process_data()

    def _load_and_process_data(self):
        with self._busy_lock:
            self.task_config = TasksConfigsDao.load_config(self.task_config.name)
            self._read_all_scraped()
            self._calc_duplicates()
            self._set_df_recent()
        if len(self.df_recent):
            self._rank_jobs()

    def rerank_jobs(self, background=False):
        if background:
            self._do_in_background(self._rank_jobs)
        else:
            self._rank_jobs()

    def _rank_jobs(self):
        with self._busy_lock:
            self.task_config = TasksConfigsDao.load_config(self.task_config.name)
            self.df_recent = self._add_model_score(self.df_recent)
            self.df_recent = self._sort_jobs(self.df_recent)
            self._unlabeled = None

    def _read_all_scraped(self):
        files = CrawlsFilesDao.get_crawls(
            self.task_config, raise_on_missing=True)

        df_all = pd.concat(
            [CrawlsFilesDao.read_scrapy_file(file) for file in files],
            axis=0, sort=False). \
            dropna(subset=[self.description_col])

        # basic deduping by url for all-read jobs
        self.df_all_read = df_all.drop_duplicates(
            subset=['url'], keep='last')

    def _calc_duplicates(self):

        df_all = self.df_all_read

        keep_inds, dup_dict_inds = calc_duplicates(
            df_all[self.description_col], keep='last')

        urls = df_all['url'].values
        self.dup_dict = {urls[i]: urls[sorted([i] + list(dups))]
                         for i, dups in dup_dict_inds.items()}

        # dedup by content and keep last
        self.df_all_deduped = df_all.iloc[keep_inds]

        logger.info(f'total historic jobs DF: {len(self.df_all_deduped)} '
                    f'(deduped from {len(df_all)})')

    def _set_df_recent(self):
        self.recent_crawl_source = CrawlsFilesDao.get_crawls(
            task_config=self.task_config)[-1]
        recent_full_df = CrawlsFilesDao.read_scrapy_file(self.recent_crawl_source)
        if not self.dedup_recent:
            self.df_recent = recent_full_df
            self._add_duplicates_column()
        else:
            self.df_recent = (
                self.df_all_deduped.loc[self.df_all_deduped['scraped_file'] ==
                                        self.recent_crawl_source, :])
            unlabeled = [u for u in self.df_recent['url']
                         if not self.labeler.is_labeled(u)]
            self.df_recent = self.df_recent[self.df_recent['url'].isin(unlabeled)]

        logger.info(f'most recent scrape DF: '
                    f'{len(self.df_recent)} ({self.recent_crawl_source}, '
                    f'all scraped: {len(recent_full_df)})')

    def _add_duplicates_column(self):
        dup_no_self = {k: [u for u in v if u != k]
                       for k, v in self.dup_dict.items()}
        dups_lists = [(l if l else None) for l in list(dup_no_self.values())]
        df_dups = pd.DataFrame({
            'url': list(dup_no_self.keys()),
            'duplicates': dups_lists,
            # 'duplicates_avg_label':
        })
        self.df_recent = pd.merge(self.df_recent, df_dups, on='url', how='left')

    def url_data(self, url):
        not_show_cols = (
                [self.description_col, 'raw_description', 'description_length',
                 'scraped_file', 'salary', 'date', 'search_url'] +
                self.intermidiate_score_cols)
        row = self.df_recent.loc[self.df_recent['url'] == url].iloc[0]
        row_disp = row.drop(not_show_cols, errors='ignore').dropna()
        row_disp = row_disp.loc[~row_disp.astype(str).isin(['0', '0.0', '[]'])]
        raw_description = row['raw_description'] if 'raw_description' in row.index else ''
        return row_disp, raw_description

    def _unlabeled_gen(self):
        urls = self.df_recent['url'].tolist()
        for url in urls:
            if not self.labeler.is_labeled(url):
                yield url

    def next_unlabeled(self):
        if self._unlabeled is None:
            self._unlabeled = self._unlabeled_gen()
        return next(self._unlabeled, None)

    def _sort_jobs(self, df):
        sort_cols = [self.model_score_col,
                     self.keyword_score_col,
                     self.scrape_order_rank_col]
        scores = [self.model_score,
                  self.keyword_score,
                  self.scrape_order_score]

        if not any(scores):
            return df  # didn't train

        self.sort_col = sort_cols[int(numpy.nanargmax(scores))]

        logger.info(f'Sorting by column: {self.sort_col} ({self.ranking_scores})')
        df.sort_values(self.sort_col, ascending=False, inplace=True)
        return df

    @property
    def ranking_scores(self):
        return (f'model-score = {self.model_score:.2f}, '
                f'keyword-score = {self.keyword_score:.2f}, '
                f'scrape-order-score = {self.scrape_order_score:.2f}')

    @classmethod
    def _extract_numeric_fields(cls, df):

        if not all(col in df.columns
                   for col in ['days_age', 'salary_low', 'salary_high']):
            df = df.apply(_extract_numeric_fields_on_row, axis=1)

        if not cls.years_experience_col in df.columns:
            df = _extract_year_experience(df, col_name=cls.years_experience_col)

        return df

    def _add_keyword_features(self, df):

        def group_named_regex(word_list, name):
            return '(?P<%s>' % name + '|'.join(word_list) + ')'

        for source, weight in product([self.description_col, self.title_col],
                                      ['positive', 'negative']):

            group_kind = f'{source}_{weight}'
            group_col = group_kind + '_count'
            group_hits = group_kind + '_hits'
            keywords = self.task_config[group_kind]
            group_regex = re.compile(group_named_regex(keywords, group_kind))

            df[group_hits] = df[source].apply(lambda s: numpy.unique(re.findall(group_regex, s)))
            # df[group_hits] = df[source].str.extractall(group_regex).groupby(level=0).agg(set)  # alternative impl
            df[group_col] = df[source].str.count(group_regex)

            if group_col not in self.intermidiate_score_cols:
                self.intermidiate_score_cols.append(group_col)

        rank_params = dict(pct=True, ascending=True)

        df[self.keyword_score_col] = (
                df.description_positive_count.rank(**rank_params)
                + df.title_positive_count.rank(**rank_params)
                - df.description_negative_count.rank(**rank_params)
                - df.title_negative_count.rank(**rank_params))

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
        df_train = self.df_all_deduped.copy()
        df_train = self._add_salary_features(df_train)

        target_col = 'salary_high'

        df_train.dropna(subset=self.text_cols + [target_col], inplace=True)
        logger.info(f'training with {len(df_train)} salaries')

        if len(df_train) >= common.MLParams.min_training_samples:
            self.regressor_salary = regression.LGBPipeline(
                text_cols=self.text_cols, num_cols=self.num_cols_salary)
            model_metrics, _ = self.regressor_salary.train_eval(
                df_train, y_col=target_col, target_name='salary')

            self.reg_sal_model_score = model_metrics[regression.MAIN_METRIC]

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

        deduped_labels = self.labeler.export_df(dedup=True)

        # using df_all_read with duplicates because can't know which duplicate was labeled
        # but since labeled df will be deduped - we'll have no dups after join
        df_train = deduped_labels.set_index('url'). \
            join(self.df_all_read.set_index('url'), how='left')

        df_train[self.target_col] = df_train[self.labeler.label_col]

        return df_train

    def _train_label_regressor(self):

        df_train = self._train_df_with_labels()

        df_train.dropna(subset=self.text_cols, inplace=True)

        logger.info(f'training with {len(df_train)} labels out of {self.labeler} '
                    f'due to missing data (possibly due to date filtering)')

        if len(df_train) >= common.MLParams.min_training_samples:
            df_train = self._add_relevance_features(df_train)

            self.regressor = regression.LGBPipeline(
                text_cols=self.text_cols, num_cols=self.num_cols_label)

            model_metrics, baselines_metrics = self.regressor.train_eval(
                df_train,
                y_col=self.target_col,
                target_name='label',
                baselines=[df_train[self.keyword_score_col],
                           df_train[self.scrape_order_rank_col]])

            metric = regression.MAIN_METRIC
            self.model_score = model_metrics[metric]
            self.keyword_score = baselines_metrics[0][metric]
            self.scrape_order_score = baselines_metrics[1][metric]

        else:
            logger.warn(f'Not training label regressor due to '
                        f'having only {len(df_train)} samples')


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


def _extract_year_experience(df, col_name):
    # inspect regexp r'(?P<years>.{1,10}[\d+.{1,3}]?\d+\syears.{1,10})'
    years_regexp = re.compile(r'(?P<years>\d+)\s*years')
    df.loc[:, col_name] = (df['description'].str.extractall(years_regexp)
                           .groupby(level=0)['years'].apply(list)
                           .apply(lambda l: min([int(el) for el in l])))
    df.loc[df[col_name] >= 12, col_name] = 0
    df.loc[df[col_name].isna(), col_name] = 0
    return df
