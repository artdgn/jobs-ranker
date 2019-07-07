#!/usr/bin/env python

from argparse import ArgumentParser

from inputs import text
from crawler.scraping import start_scraping
from joblist.ranking import JobsRanker
from tasks.dao import TasksDao


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

    task_chooser = text.TaskChooser(tasks_dao=TasksDao())
    task_config = task_chooser.load_or_choose_task(task_name=args.task_json)

    if args.scrape:
        start_scraping(task_config=task_config,
                       http_cache=args.http_cache,
                       blocking=True)

    jobs_ranker = JobsRanker(
        task_config=task_config,
        dedup_new=(not args.no_dedup),
        skipped_as_negatives=args.assume_negative)

    jobs_ranker.load_and_process_data(background=False)

    labeler = text.Labeler(jobs_ranker=jobs_ranker)

    labeler.label_jobs_loop(recalc_everytime=args.recalc)


if __name__ == '__main__':
    main()
