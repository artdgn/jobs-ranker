import os
import datetime
import time
from multiprocessing import Process

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from crawler.spiders.css_spiders import ToScrapeCSSSpider
from joblist.joblist_processing import JobsListLabeler


crawls_dir = './data/crawls'
labeled_jobs_csv = 'data/labeled_jobs.csv'
keywords_json = 'data/keywords.json'

log_file = 'log-%s.log' % datetime.datetime.now().isoformat()
crawl_name = 'jora-%s' % datetime.datetime.now().date().isoformat()
crawl_output = os.path.join(crawls_dir, '%s.csv' % crawl_name)


def crawl():
    settings = get_project_settings()
    settings.set('FEED_FORMAT', 'csv', priority='cmdline')
    settings.set('FEED_URI', crawl_output, priority='cmdline')
    settings.set('JOBDIR', './crawler/crawls/%s' % crawl_name, priority='cmdline')
    settings.set('LOG_FILE', './data/%s' % log_file, priority='cmdline')
    # settings.set('LOG_STDOUT', True, priority='cmdline')

    process = CrawlerProcess(settings)
    process.crawl(ToScrapeCSSSpider)
    process.start()

Process(target=crawl).start()

while not os.path.exists(crawl_output):
    print('waiting for output..')
    time.sleep(2)

crawls = [os.path.join(crawls_dir, f) for f in os.listdir(crawls_dir)]
crawls.remove(crawl_output)


jobs = JobsListLabeler(
    scraped=crawl_output,
    keywords=keywords_json,
    labeled=labeled_jobs_csv,
    older_scraped=crawls)

jobs.label_jobs()







