import os

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

import crawler.settings
from crawler.jora_scraper import get_jora_spider_for_url

from tasks.config import TaskConfig

import common


def start_scraping(task_config: TaskConfig, http_cache=False):
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
    return process, log_path
