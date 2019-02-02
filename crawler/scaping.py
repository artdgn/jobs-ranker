import datetime
import os

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from crawler.spiders.css_spiders import ToScrapeCSSSpider

crawls_dir = './data/crawls'


def start_scraping():
    log_file = 'log-%s.log' % datetime.datetime.now().isoformat()
    log_path = os.path.abspath('./data/logs/%s' % log_file)
    crawl_name = 'jora-%s' % datetime.datetime.now().date().isoformat()
    crawl_output = os.path.join(crawls_dir, '%s.csv' % crawl_name)

    settings = get_project_settings()
    settings.set('FEED_FORMAT', 'csv', priority='cmdline')
    settings.set('FEED_URI', crawl_output, priority='cmdline')
    settings.set('JOBDIR', './crawler/crawls/%s' % crawl_name, priority='cmdline')
    settings.set('LOG_FILE', log_path, priority='cmdline')
    # settings.set('LOG_STDOUT', True, priority='cmdline')

    process = CrawlerProcess(settings)
    process.crawl(ToScrapeCSSSpider)
    process.start()
    return process, log_path