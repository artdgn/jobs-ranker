import os
import time
from multiprocessing import Process
from argparse import ArgumentParser

from crawler.scaping import start_scraping, crawls_dir
from joblist.joblist_processing import JobsListLabeler

labeled_jobs_csv = 'data/labeled_jobs.csv'
keywords_json = 'data/keywords.json'


parser = ArgumentParser()
parser.add_argument("-s", "--scrape", action="store_true",
                    help="whether to scrape")
parser.add_argument("-d", "--delay", type=float, default=10,
                    help="delay in seconds between after start of scraping")
args = parser.parse_args()


if args.scrape:
    s_proc = Process(target=start_scraping).start()
    print(f'Starting scraping.. waiting for {args.delay} '
          f'seconds before labeling results.')
    time.sleep(args.delay)

crawls = [os.path.join(crawls_dir, f) for f in sorted(os.listdir(crawls_dir))]

jobs = JobsListLabeler(
    scraped=crawls.pop(),
    keywords=keywords_json,
    labeled=labeled_jobs_csv,
    older_scraped=crawls)

jobs.label_jobs()

