import os
import subprocess

import pandas as pd
import pandas.errors

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from jobs_recommender.crawler.jora_scraper import JoraSpider
from jobs_recommender.crawler import settings as crawler_settings

from jobs_recommender.tasks.config import TaskConfig

from jobs_recommender import common
from jobs_recommender.utils import logger


class CrawlProcess:

    def __init__(self, task_config: TaskConfig, http_cache=False):
        self.task_config = task_config
        self.http_cache = http_cache
        crawl_name = f'jora-{common.CURRENT_DATE}'

        self.log_path = os.path.join(
            self.task_config.scrapy_log_dir,
            f'log-{common.CURRENT_TIMESTAMP}.log')

        self.crawl_output_path = os.path.join(
            self.task_config.crawls_dir, f'{crawl_name}.csv')

        self.jobdir_path = os.path.join(
            self.task_config.crawl_job_dir, crawl_name)

    def _settings(self):
        os.environ['SCRAPY_SETTINGS_MODULE'] = crawler_settings.__name__
        settings = get_project_settings()
        settings.set('FEED_FORMAT', 'csv', priority='cmdline')
        settings.set('FEED_URI', self.crawl_output_path, priority='cmdline')
        settings.set('JOBDIR', self.jobdir_path, priority = 'cmdline')
        settings.set('LOG_FILE', self.log_path, priority='cmdline')
        settings.set('HTTPCACHE_ENABLED', self.http_cache, priority='cmdline')
        return settings

    def start_scraping(self):
        self.proc = CrawlerProcess(settings=self._settings())
        self.proc.crawl(JoraSpider(start_urls=self.task_config.search_url))
        self.proc.start()

        logger.info(f'Started scraping task "{self.task_config.name}", '
                    f'check log file at {self.log_path}, '
                    f'output file at {self.crawl_output_path}')

    def start_subprocess(self):
        subprocess.Popen(['scrapy', 'crawl'])

    def join(self):
        self.proc.join()

class CrawlsFilesDao:

    @staticmethod
    def read_scrapy_file(filename):
        try:
            df = pd.read_csv(filename)
        except pandas.errors.EmptyDataError:
            logger.info(f'found empty scrape file:{filename}. '
                        f'trying to delete.')
            os.remove(filename)
            return pd.DataFrame()
        else:
            drop_cols = ([col for col in df.columns
                          if col.startswith('download_')] + ['depth'])
            df.drop(drop_cols, axis=1, inplace=True)
            df['scraped_file'] = filename
            return df

    @staticmethod
    def all_crawls(task_config: TaskConfig, raise_on_missing=True):
        all_crawls = [os.path.join(task_config.crawls_dir, f)
                      for f in sorted(os.listdir(task_config.crawls_dir))]
        if raise_on_missing and not all_crawls:
            raise FileNotFoundError(
                f'No crawls found for task "{task_config.name}", '
                f'please run scraping.')
        return all_crawls

    @classmethod
    def days_since_last_crawl(cls, task_config: TaskConfig):
        latest = cls.all_crawls(task_config)[-1]
        date = os.path.splitext(latest)[0][-10:]
        return (pd.to_datetime(common.CURRENT_DATE) -
                pd.to_datetime(date)).days
