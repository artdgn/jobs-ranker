import os
from argparse import ArgumentParser

from crawler.scaping import start_scraping, crawls_dir
from joblist.joblist_processing import JobsListLabeler

labeled_jobs_csv = 'data/labeled_jobs.csv'
keywords_json = 'data/keywords.json'


parser = ArgumentParser()
parser.add_argument("-s", "--scrape", action="store_true",
                    help="whether to scrape")
parser.add_argument("-r", "--recalc", action="store_true",
                    help="whether to recalc model after every new label")
parser.add_argument("-n", "--no-dedup", action="store_true",
                    help="prevent deduplication of newest scrapes w/r to historic scrapes")
parser.add_argument("-a", "--assume-negative", action="store_true",
                    help="assume jobs left unlabeled previously as labeled negative")
args = parser.parse_args()

os.chdir(os.path.realpath(os.path.dirname(__file__)))


if args.scrape:
    s_proc, scrape_log = start_scraping()
    print(f'Started scraping, waiting for results... check log file at {scrape_log}')
    s_proc.join()


crawls = [os.path.join(crawls_dir, f)
          for f in sorted(os.listdir(crawls_dir))]

jobs = JobsListLabeler(
    scraped=crawls.pop(),
    keywords=keywords_json,
    labeled=labeled_jobs_csv,
    older_scraped=crawls,
    dedup_new=(not args.no_dedup),
    skipped_as_negatives=args.assume_negative)

jobs.label_jobs(recalc_everytime=args.recalc)

