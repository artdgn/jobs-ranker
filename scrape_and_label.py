#!/usr/bin/env python

import os
from argparse import ArgumentParser

from crawler.scraping import start_scraping
from joblist.joblist_processing import JobsListLabeler

from tasks.config import get_task_config
from utils.logger import logger


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-t", "--task-json", type=str, default="",
                        help="path to json file or task name with file in the ./data dir "
                             "that contains the task configuration")
    parser.add_argument("-s", "--scrape", action="store_true",
                        help="whether to scrape. default false")
    parser.add_argument("-c", "--http-cache", action="store_true",
                        help="whether to use http cache (helpful for debugging scraping). default false")
    parser.add_argument("-r", "--recalc", action="store_true",
                        help="whether to recalc model after every new label. default false")
    parser.add_argument("-n", "--no-dedup", action="store_true",
                        help="prevent deduplication of newest scrapes w/r to historic scrapes. default false")
    parser.add_argument("-a", "--assume-negative", action="store_true",
                        help="assume jobs left unlabeled previously as labeled negative. default false")
    return parser.parse_args()


def main():
    args = parse_args()

    os.chdir(os.path.realpath(os.path.dirname(__file__)))

    task_config = get_task_config(task_name=args.task_json)

    if args.scrape:
        s_proc, scrape_log = start_scraping(task_config=task_config,
                                            http_cache=args.http_cache)
        logger.info(f'Started scraping, waiting for results... '
                    f'check log file at {scrape_log}')
        s_proc.join()

    crawls = [os.path.join(task_config.crawls_dir, f)
              for f in sorted(os.listdir(task_config.crawls_dir))]

    if crawls:
        jobs = JobsListLabeler(
            scraped=crawls.pop(),
            task_config=task_config,
            older_scraped=crawls,
            dedup_new=(not args.no_dedup),
            skipped_as_negatives=args.assume_negative)

        jobs.label_jobs(recalc_everytime=args.recalc)

    else:
        logger.info('No scraped jobs found, please run scraping before '
                    'labeling by specifying -s (--scrape) flag')


if __name__ == '__main__':
    main()