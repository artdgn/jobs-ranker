import os

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from crawler.spiders.css_spiders import ToScrapeCSSSpider

import common


def start_scraping(http_cache=False):
    log_path = os.path.join(common.SCRAPY_LOG_DIR, f'log-{common.CURRENT_TIMESTAMP}.log')
    crawl_name = f'jora-{common.CURRENT_DATE}'
    crawl_output = os.path.join(common.CRAWLS_DIR, f'{crawl_name}.csv')

    settings = get_project_settings()
    settings.set('FEED_FORMAT', 'csv', priority='cmdline')
    settings.set('FEED_URI', crawl_output, priority='cmdline')
    settings.set('JOBDIR', os.path.join(common.CRAWLS_JOB_DIR, crawl_name), priority='cmdline')
    settings.set('LOG_FILE', log_path, priority='cmdline')
    settings.set('HTTPCACHE_ENABLED', http_cache, priority='cmdline')

    process = CrawlerProcess(settings)
    process.crawl(ToScrapeCSSSpider)
    process.start()
    return process, log_path