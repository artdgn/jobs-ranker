import os
import time
from multiprocessing import Process

from crawler.scaping import start_scraping, crawls_dir
from joblist.joblist_processing import JobsListLabeler

labeled_jobs_csv = 'data/labeled_jobs.csv'
keywords_json = 'data/keywords.json'

# scrape = True
scrape = False

if scrape:
    s_proc = Process(target=start_scraping).start()
    print('Starting scraping.. waiting for results')
    time.sleep(10)


crawls = [os.path.join(crawls_dir, f) for f in sorted(os.listdir(crawls_dir))]

jobs = JobsListLabeler(
    scraped=crawls.pop(),
    keywords=keywords_json,
    labeled=labeled_jobs_csv,
    older_scraped=crawls)

jobs.label_jobs()

