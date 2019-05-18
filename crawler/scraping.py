import os

import pandas as pd
import pandas.errors

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

import crawler.settings
from crawler.jora_scraper import get_jora_spider_for_url

from tasks.config import TaskConfig

import common
from utils.logger import logger


def start_scraping(task_config: TaskConfig, http_cache=False, blocking=True):
    log_path = os.path.join(task_config.scrapy_log_dir,
                            f'log-{common.CURRENT_TIMESTAMP}.log')

    crawl_name = f'jora-{common.CURRENT_DATE}'
    crawl_output = os.path.join(task_config.crawls_dir, f'{crawl_name}.csv')

    os.environ['SCRAPY_SETTINGS_MODULE'] = crawler.settings.__name__
    settings = get_project_settings()
    settings.set('FEED_FORMAT', 'csv', priority='cmdline')
    settings.set('FEED_URI', crawl_output, priority='cmdline')
    settings.set('JOBDIR',
                 os.path.join(task_config.crawl_job_dir, crawl_name),
                 priority='cmdline')
    settings.set('LOG_FILE', log_path, priority='cmdline')
    settings.set('HTTPCACHE_ENABLED', http_cache, priority='cmdline')

    process = CrawlerProcess(settings)
    process.crawl(get_jora_spider_for_url(search_url=task_config.search_url))
    process.start()

    if blocking:
        logger.info(f'Started scraping, waiting for results... '
                    f'check log file at {log_path}')
        process.join()
    else:
        return process


def read_scrapy_file(filename):
    try:
        df = pd.read_csv(filename)
    except pandas.errors.EmptyDataError:
        logger.info(f'found empty scrape file:{filename}. trying to delete.')
        os.remove(filename)
        return pd.DataFrame()
    else:
        drop_cols = ([col for col in df.columns if col.startswith('download_')]
                     + ['depth'])
        df.drop(drop_cols, axis=1, inplace=True)
        df['scraped_file'] = filename
        return df